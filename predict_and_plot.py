import os, json, math, glob, shutil, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime, timedelta
from tqdm import tqdm
import requests

warnings.filterwarnings("ignore")

# =============================================================================
# 1. 設定
# =============================================================================

MODEL_URL     = "https://huggingface.co/EGTY/Weather_model_V4.5/resolve/main/best_model_3D_optimized%20(6).pth"
MODEL_PATH    = "best_model_3D_optimized (6).pth"
TEST_NPZ_PATH = "V4_model_data.npz"
STATIC_PATHS  = [
    "environment-2down/land_mask.npy",
    "environment-2down/soil_type.npy",
    "environment-2down/topography.npy",
]

VAR_NAMES = [
    't2m', 'msl', 'u10', 'v10', 'sst',
    'z_850', 'u_850', 'v_850', 't_850',
    'z_500', 'u_500', 'v_500', 'q_500',
    'u_200', 'v_200',
]

FIXED_STATS = {
    't2m':   {'mean': 280.0,    'std': 25.0},
    'msl':   {'mean': 101000.0, 'std': 1400.0},
    'u10':   {'mean': 0.0,      'std': 10.0},
    'v10':   {'mean': 0.0,      'std': 10.0},
    'sst':   {'mean': 287.0,    'std': 12.0},
    'z_850': {'mean': 14000.0,  'std': 1600.0},
    'u_850': {'mean': 1.0,      'std': 10.0},
    'v_850': {'mean': 0.0,      'std': 7.0},
    't_850': {'mean': 275.0,    'std': 16.0},
    'z_500': {'mean': 55000.0,  'std': 3500.0},
    'u_500': {'mean': 7.0,      'std': 15.0},
    'v_500': {'mean': 0.0,      'std': 15.0},
    'q_500': {'mean': 0.0009,   'std': 0.0012},
    'u_200': {'mean': 12.0,     'std': 20.0},
    'v_200': {'mean': 0.0,      'std': 15.0},
}

SEQ_LEN                      = 2
BASE_CH                      = 128
GROUP_NORM_GROUPS            = 8
AUTOREGRESSIVE_STEPS         = 30
PERTURBED_AUTOREGRESSIVE_STEPS = 30
DEVICE                       = "cuda" if torch.cuda.is_available() else "cpu"

# 追蹤參數
DEFAULT_SEARCH_RADIUS_KM    = 150
VORTEX_VALIDATION_RADIUS_KM = 200
PATIENCE_RADIUS_INCREASE_KM = 50
PATIENCE_WINDOW_HOURS       = 24
WIND_R34_THRESHOLD_MS       = 34 * 0.514444   # 17.49 m/s
MAX_WIND_RADIUS_SEARCH_KM   = 500

# 微擾強度（標準化空間中的加成噪聲）
PERTURBATION_STD = 0.2

VIS_VARS = ['msl', 't2m', 'q_500', 'wind_speed']

REGIONS = {
    'global':        {'name': 'Global',       'extent': None},
    'west_pacific':  {'name': 'West Pacific',  'extent': [90,  180,   0,  45]},
    'south_pacific': {'name': 'South Pacific', 'extent': [90,  180, -45,   0]},
    'east_pacific':  {'name': 'East Pacific',  'extent': [-180, -90,   0,  45]},
    'east_atlantic': {'name': 'East Atlantic', 'extent': [-90,    0,   0,  45]},
}

# 色條
q_cmap = mcolors.LinearSegmentedColormap.from_list(
    "custom_diverging", ["#613A00", '#FFFFFF', "#008F92", "#004792"])
wind_cmap = mcolors.LinearSegmentedColormap.from_list(
    "custom_diverging", ["#FFFFFF", "#00C4B3", "#03C400", "#C4C400",
                         "#C45800", "#C40000", "#C40093"])

VAR_PLOT = {
    't2m':       {'long_name': '2m Temperature',
                  'unit': '°C',  'cmap': 'RdYlBu_r',  'symmetric': False,
                  'contour': True, 'n_levels': 15, 'vmax':  40.0, 'vmin': -60.0},
    'msl':       {'long_name': 'Mean Sea Level Pressure',
                  'unit': 'hPa', 'cmap': 'Spectral_r', 'symmetric': False,
                  'contour': True, 'n_levels': 20, 'vmax': 1072.0, 'vmin': 952.0},
    'q_500':     {'long_name': '500 hPa Specific Humidity with MSLP Isobars',
                  'unit': 'g/kg', 'cmap': q_cmap, 'symmetric': False,
                  'contour': True, 'n_levels': -0.5, 'vmax': 5, 'vmin': 0},
    'wind_speed': {'long_name': '10m Wind Speed with MSLP Isobars',
                   'unit': 'm/s', 'cmap': wind_cmap, 'symmetric': False,
                   'contour': True, 'n_levels': 0, 'vmax': 30.0, 'vmin': 0},
}

GIF          = False
GIF_DURATION = 0.5
SEED         = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# =============================================================================
# 2. 模型定義
# =============================================================================

def _custom_polar_padding(x, pad_y, vector_indices=None):
    if pad_y == 0:
        return x
    north_src = x[:, :, :, :pad_y, :]
    south_src = x[:, :, :, -pad_y:, :]
    if vector_indices:
        north_src = north_src.clone(); north_src[:, vector_indices] *= -1.0
        south_src = south_src.clone(); south_src[:, vector_indices] *= -1.0
    W = x.shape[-1]
    north = torch.flip(torch.roll(north_src, shifts=W // 2, dims=-1), dims=[-2])
    south = torch.flip(torch.roll(south_src, shifts=W // 2, dims=-1), dims=[-2])
    return torch.cat([north, x, south], dim=-2)


class CircularConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 bias=True, vector_indices=None):
        super().__init__()
        ks = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 3
        self.kernel_size    = ks
        self.vector_indices = vector_indices
        self.conv = nn.Conv3d(in_channels, out_channels, ks, stride, padding=0, bias=bias)

    def forward(self, x):
        pd, py, px = [(k - 1) // 2 for k in self.kernel_size]
        x = _custom_polar_padding(x, py, self.vector_indices)
        if px > 0:
            x = torch.cat([x[..., -px:], x, x[..., :px]], dim=-1)
        if pd > 0:
            x = F.pad(x, (0, 0, 0, 0, pd, pd), 'reflect')
        return self.conv(x)


class CircularConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, groups=8, is_first_layer=False,
                 vector_indices=None):
        super().__init__()
        fk = (2, 3, 3) if is_first_layer else (1, 3, 3)
        self.net = nn.Sequential(
            CircularConv3d(in_ch,  out_ch, fk,      vector_indices=vector_indices),
            nn.GroupNorm(groups, out_ch), nn.ReLU(True),
            CircularConv3d(out_ch, out_ch, (1,3,3), vector_indices=None),
            nn.GroupNorm(groups, out_ch), nn.ReLU(True),
        )

    def forward(self, x):
        return self.net(x)


class CircularUNet3D(nn.Module):
    def __init__(self, in_ch, base_ch, out_ch, groups, vector_channel_indices=None):
        super().__init__()
        B = base_ch
        self.enc1  = CircularConvBlock3D(in_ch, B,    groups, is_first_layer=True,
                                         vector_indices=vector_channel_indices)
        self.pool1 = nn.MaxPool3d((1, 2, 2))
        self.enc2  = CircularConvBlock3D(B,   B*2, groups)
        self.pool2 = nn.MaxPool3d((1, 2, 2))
        self.enc3  = CircularConvBlock3D(B*2, B*4, groups)
        self.pool3 = nn.MaxPool3d((1, 2, 2))
        self.enc4  = CircularConvBlock3D(B*4, B*8, groups)
        self.b     = CircularConvBlock3D(B*8, B*8, groups)
        self.u3    = nn.ConvTranspose3d(B*8, B*4, (1,2,2), stride=(1,2,2))
        self.dec3  = CircularConvBlock3D(B*8, B*4, groups)
        self.u2    = nn.ConvTranspose3d(B*4, B*2, (1,2,2), stride=(1,2,2))
        self.dec2  = CircularConvBlock3D(B*4, B*2, groups)
        self.u1    = nn.ConvTranspose3d(B*2, B,   (1,2,2), stride=(1,2,2))
        self.dec1  = CircularConvBlock3D(B*2, B,   groups)
        self.final = nn.Conv3d(B, out_ch, 1)

    def forward(self, x):
        c1 = self.enc1(x);  p1 = self.pool1(c1)
        c2 = self.enc2(p1); p2 = self.pool2(c2)
        c3 = self.enc3(p2); p3 = self.pool3(c3)
        c4 = self.enc4(p3)
        b  = self.b(c4)
        d3 = self.dec3(torch.cat([c3, self.u3(b)],  dim=1))
        d2 = self.dec2(torch.cat([c2, self.u2(d3)], dim=1))
        d1 = self.dec1(torch.cat([c1, self.u1(d2)], dim=1))
        return self.final(d1).squeeze(2)

# =============================================================================
# 3. 資料處理工具
# =============================================================================

def load_static(paths):
    tensors = []
    for p in paths:
        if not os.path.exists(p):
            print(f"[警告] 靜態檔案不存在: {p}")
            continue
        arr = np.load(p).astype(np.float32)
        t   = torch.from_numpy(arr)
        if t.ndim == 2:
            t = t.unsqueeze(0)
        tensors.append(t)
    return torch.cat(tensors, dim=0) if tensors else None


def destandardize(pred_std, means, stds):
    """反標準化 (T, C, H, W) 並做單位換算。"""
    m    = np.array(means, dtype=np.float32).reshape(1, -1, 1, 1)
    s    = np.array(stds,  dtype=np.float32).reshape(1, -1, 1, 1)
    data = pred_std * s + m
    for c, v in enumerate(VAR_NAMES):
        if v == 't2m':    data[:, c] -= 273.15
        elif v == 'msl':  data[:, c] /= 100.0
        elif v == 'q_500':data[:, c] *= 1000.0
    return data


def convert_numpy(obj):
    """遞迴將 numpy 純量轉為 Python 基本型別（JSON 序列化用）。"""
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj

# =============================================================================
# 4. 自迴歸預測
# =============================================================================

@torch.no_grad()
def autoregressive_predict(model, init_seq_std, static_tensor, steps, device,
                            desc="自迴歸預測"):
    model.eval()
    T, _, H, W = init_seq_std.shape
    seq_np     = init_seq_std.copy()
    preds      = []

    static_gpu = static_tensor.to(device) if static_tensor is not None else None
    static_exp = (static_gpu.unsqueeze(0).expand(T, -1, -1, -1)
                  if static_gpu is not None else None)

    for _ in tqdm(range(steps), desc=desc):
        wx = torch.from_numpy(seq_np).float().to(device)
        x  = torch.cat([wx, static_exp], dim=1) if static_exp is not None else wx
        x  = x.permute(1, 0, 2, 3).unsqueeze(0)

        with torch.amp.autocast(device_type=device, enabled=(device == "cuda")):
            out = model(x)

        out_np     = out.squeeze(0).cpu().numpy()
        preds.append(out_np)
        seq_np     = np.roll(seq_np, -1, axis=0)
        seq_np[-1] = out_np

    return np.stack(preds, axis=0)


def perturb_init_seq(init_seq_std, std=PERTURBATION_STD, seed=SEED + 1):
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(init_seq_std.shape).astype(np.float32) * std
    return init_seq_std + noise

# =============================================================================
# 5. 地理計算工具
# =============================================================================

def haversine(lon1, lat1, lon2, lat2):
    R = 6371.0
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def coords_to_grid(lat, lon, H, W):
    lon360 = lon if lon >= 0 else lon + 360
    i = int(round((90.0 - lat) / (180.0 / (H - 1))))
    j = int(round(lon360 / (360.0 / W))) % W
    return min(max(i, 0), H - 1), j


def grid_to_coords(i, j, H, W):
    lat = 90.0 - i * (180.0 / (H - 1))
    lon = j * (360.0 / W)
    return lat, (lon if lon <= 180 else lon - 360)

# =============================================================================
# 6. 颱風追蹤核心
# =============================================================================

def _var_idx(name):
    return VAR_NAMES.index(name)


def wind_speed_at(field, i, j):
    u = field[_var_idx('u10'), i, j]
    v = field[_var_idx('v10'), i, j]
    return math.sqrt(u*u + v*v)


def validate_cyclonic_circulation(field, center_lat, center_lon):
    _, H, W = field.shape
    u_vals, v_vals, max_ws = [], [], 0.0
    for i in range(H):
        for j in range(W):
            lat, lon = grid_to_coords(i, j, H, W)
            if haversine(center_lon, center_lat, lon, lat) <= VORTEX_VALIDATION_RADIUS_KM:
                u = field[_var_idx('u10'), i, j]
                v = field[_var_idx('v10'), i, j]
                u_vals.append(u); v_vals.append(v)
                max_ws = max(max_ws, math.sqrt(u*u + v*v))
    if not u_vals:
        return False
    return (max_ws > 6
            and max(u_vals) - min(u_vals) > 5
            and max(v_vals) - min(v_vals) > 5)


def get_max_wind_speed(field, center_lat, center_lon,
                        search_km=MAX_WIND_RADIUS_SEARCH_KM):
    _, H, W = field.shape
    max_ws  = 0.0
    for i in range(H):
        for j in range(W):
            lat, lon = grid_to_coords(i, j, H, W)
            if haversine(center_lon, center_lat, lon, lat) <= search_km:
                max_ws = max(max_ws, wind_speed_at(field, i, j))
    return max_ws


def compute_wind_radii(field, center_lat, center_lon):
    """計算四象限 34 kt 風圈半徑（km）。"""
    _, H, W = field.shape
    quadrant_bounds = {'NE': (0, 90), 'SE': (90, 180),
                       'SW': (180, 270), 'NW': (270, 360)}
    result = {}
    radial_bins = np.arange(10, MAX_WIND_RADIUS_SEARCH_KM + 10, 10)

    for quad, (a_min, a_max) in quadrant_bounds.items():
        found, last_valid, fail_tol = False, 0.0, 0
        for radius in radial_bins:
            hit = total = 0
            for angle in range(0, 360, 5):
                if not (a_min <= angle < a_max):
                    continue
                total += 1
                rad  = math.radians(angle)
                dlat = (radius / 111.0) * math.cos(rad)
                dlon = (radius / (111.0 * max(0.15, abs(math.cos(math.radians(center_lat)))))) * math.sin(rad)
                i, j = coords_to_grid(center_lat + dlat, center_lon + dlon, H, W)
                if wind_speed_at(field, i, j) >= WIND_R34_THRESHOLD_MS:
                    hit += 1
            ring_valid = (hit / max(total, 1)) >= 0.15
            if not found:
                if ring_valid:
                    found, last_valid, fail_tol = True, radius, 0
                continue
            if ring_valid:
                last_valid, fail_tol = radius, 0
            else:
                fail_tol += 1
                if fail_tol >= 2:
                    break
        result[quad] = last_valid
    return result


def find_pressure_minimum(field, start_lat, start_lon, search_km):
    _, H, W    = field.shape
    msl_idx    = _var_idx('msl')
    min_p, best = float('inf'), None
    for i in range(H):
        for j in range(W):
            lat, lon = grid_to_coords(i, j, H, W)
            if haversine(start_lon, start_lat, lon, lat) <= search_km:
                p = field[msl_idx, i, j]
                if p < min_p:
                    min_p, best = p, (i, j)
    return best, min_p


def _build_intensity(field, center_lat, center_lon, mslp):
    max_wind = get_max_wind_speed(field, center_lat, center_lon)
    radii    = compute_wind_radii(field, center_lat, center_lon)
    return {
        "mslp_hpa":        float(mslp),
        "max_wind_knots":  float(max_wind * 1.94384),
        "r34_quadrant_km": radii,
    }


def _build_init_point(field, ty, base_time):
    """建立 t=0 觀測初始點。"""
    radii = compute_wind_radii(field, float(ty['latitude']), float(ty['longitude']))
    return {
        "valid_time":  base_time.strftime("%Y-%m-%d %H:%M:%S"),
        "coordinates": {"lat": float(ty['latitude']), "lon": float(ty['longitude'])},
        "intensity": {
            "mslp_hpa":        float(ty['mslp']),
            "max_wind_knots":  float(ty['wind']),
            "r34_quadrant_km": radii,
        },
    }


def track_typhoon(preds_phys, init_track, ty_id, base_time, desc_prefix=""):
    """
    追蹤單一颱風，回傳預測後的各時間點列表。
    init_track: list of dicts，至少包含一個初始點。
    """
    steps, _, H, W = preds_phys.shape
    full_track     = list(init_track)

    in_patience      = False
    patience_radius  = 0

    for t in range(steps):
        hours       = (t + 1) * 6
        valid_time  = base_time + timedelta(hours=hours)
        prev        = full_track[-1]['coordinates']
        prev_lat, prev_lon = prev['lat'], prev['lon']

        if in_patience:
            search_km = patience_radius
            patience_radius += PATIENCE_RADIUS_INCREASE_KM
        else:
            if len(full_track) < 2:
                search_km = DEFAULT_SEARCH_RADIUS_KM
            else:
                l1, l2    = full_track[-2]['coordinates'], full_track[-1]['coordinates']
                move_dist = haversine(l1['lon'], l1['lat'], l2['lon'], l2['lat'])
                search_km = max(100, move_dist * 1.5)

        best_ij, min_p = find_pressure_minimum(
            preds_phys[t], prev_lat, prev_lon, search_km)

        valid = False
        if best_ij is not None and min_p < 1010:
            clat, clon = grid_to_coords(best_ij[0], best_ij[1], H, W)
            if validate_cyclonic_circulation(preds_phys[t], clat, clon):
                valid = True

        if valid:
            if in_patience:
                in_patience = False
            intensity = _build_intensity(preds_phys[t], clat, clon, min_p)
            full_track.append({
                "valid_time":  valid_time.strftime("%Y-%m-%d %H:%M:%S"),
                "coordinates": {"lat": float(clat), "lon": float(clon)},
                "intensity":   intensity,
            })
        else:
            if hours <= PATIENCE_WINDOW_HOURS:
                if not in_patience:
                    in_patience     = True
                    patience_radius = search_km + PATIENCE_RADIUS_INCREASE_KM
                continue
            else:
                print(f"  [{desc_prefix} t={hours}h] 追蹤中止。")
                break

    # 回傳扣除初始點的預測部分
    return full_track[len(init_track):]

# =============================================================================
# 7. 活躍颱風讀取
# =============================================================================

def get_active_typhoon_initials(base_dir="active_typhoon"):
    typhoons = []
    if not os.path.exists(base_dir):
        print(f"[警告] 找不到資料夾: {base_dir}")
        return typhoons
    for file_path in glob.glob(os.path.join(base_dir, "*", "*.dat")):
        if not os.path.basename(file_path).lower().startswith('b'):
            continue
        try:
            with open(file_path) as f:
                lines = f.readlines()
            if not lines:
                continue
            parts    = [p.strip() for p in lines[-1].strip().split(',')]
            lat      = float(parts[6][:-1]) / 10.0 * (-1 if parts[6].endswith('S') else 1)
            raw_lon  = float(parts[7][:-1]) / 10.0 * (-1 if parts[7].endswith('W') else 1)
            lon      = raw_lon - 360 if raw_lon > 180 else raw_lon
            wind_kt  = float(parts[8]) if parts[8] else 0.0
            mslp_hpa = float(parts[9]) if parts[9] else 1013.0
            ty_name  = parts[27] if len(parts) > 27 else "Unknown"
            ty_id    = f"{parts[0]}{parts[1]}{parts[2][:4]}"
            typhoons.append({"id": ty_id, "name": ty_name,
                              "latitude": lat, "longitude": lon,
                              "mslp": mslp_hpa, "wind": wind_kt})
            print(f"活躍颱風: {ty_name} ({ty_id}) | "
                  f"Lat:{lat}, Lon:{lon} | P:{mslp_hpa}hPa, V:{wind_kt}kt")
        except Exception as e:
            print(f"[錯誤] {file_path}: {e}")
    return typhoons

# =============================================================================
# 8. 繪圖（淺色主題）
# =============================================================================

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'text.color': '#111111',
    'axes.facecolor': '#F5F8FA',  'figure.facecolor': '#FFFFFF',
    'axes.edgecolor': '#D8DDE3',  'xtick.color': '#555555',
    'ytick.color':   '#555555',   'figure.dpi': 100,
    'savefig.dpi':   100,         'savefig.facecolor': '#FFFFFF',
    'savefig.bbox':  None,        'savefig.pad_inches': 0.0,
})

_OCEAN   = cfeature.NaturalEarthFeature('physical', 'ocean',     '50m', facecolor='#E8F0E4', edgecolor='none')
_LAND    = cfeature.NaturalEarthFeature('physical', 'land',      '50m', facecolor='#E8F0E4', edgecolor='none')
_COAST   = cfeature.NaturalEarthFeature('physical', 'coastline', '50m', facecolor='none',    edgecolor="#00830B", linewidth=0.75)
_BORDERS = cfeature.NaturalEarthFeature('cultural', 'admin_0_boundary_lines_land', '50m',
                                         facecolor='none', edgecolor='#9BAAB5', linewidth=0.3, linestyle='--')


def _data_extent(lons, lats):
    dlon = abs(lons[1] - lons[0]); dlat = abs(lats[0] - lats[1])
    return [lons[0]-dlon/2, lons[-1]+dlon/2, lats[-1]-dlat/2, lats[0]+dlat/2]


def _region_stats(data_2d, lons, lats, region_ext):
    if region_ext is None:
        return float(np.nanmax(data_2d)), float(np.nanmin(data_2d))
    lon_min, lon_max, lat_min, lat_max = region_ext
    lm = (lons >= lon_min) & (lons <= lon_max)
    am = (lats >= lat_min) & (lats <= lat_max)
    arr = data_2d[np.ix_(am, lm)] if lm.any() and am.any() else data_2d
    return float(np.nanmax(arr)), float(np.nanmin(arr))


def _overlay_msl_isobars(ax, lons, lats, msl_data, region_ext, proj):
    lons_m, lats_m = np.meshgrid(lons, lats)
    sub = msl_data
    if region_ext is not None:
        lon_min, lon_max, lat_min, lat_max = region_ext
        lm, am = (lons >= lon_min) & (lons <= lon_max), (lats >= lat_min) & (lats <= lat_max)
        if lm.any() and am.any():
            sub = msl_data[np.ix_(am, lm)]
    levels = np.arange(np.floor(np.nanmin(sub)/4)*4,
                       np.ceil(np.nanmax(sub)/4)*4 + 4, 4)
    try:
        cs   = ax.contour(lons_m, lats_m, msl_data, levels=levels,
                          colors='#1A1A2E', linewidths=1.0, alpha=0.80,
                          transform=proj, zorder=8)
        lbls = ax.clabel(cs, inline=True, fontsize=6.0, fmt='%.0f', use_clabeltext=True)
        for lbl in lbls:
            lbl.set_color('#1A1A2E'); lbl.set_fontsize(6.0); lbl.set_fontweight('bold')
            lbl.set_path_effects([pe.withStroke(linewidth=2.2, foreground='white')])
    except Exception as e:
        print(f"  [等壓線警告] {e}")


def _overlay_wind_barbs(ax, lons, lats, u_data, v_data, proj):
    lons_m, lats_m = np.meshgrid(lons, lats)
    step   = 5
    li, lj = np.arange(0, len(lats), step), np.arange(0, len(lons), step)
    lo_s   = lons_m[np.ix_(li, lj)]; la_s = lats_m[np.ix_(li, lj)]
    u_s    = u_data[np.ix_(li, lj)]; v_s  = v_data[np.ix_(li, lj)]
    spd_s  = np.sqrt(u_s**2 + v_s**2)
    kw = dict(transform=proj, zorder=9, length=5, linewidth=0.7, pivot='middle',
              sizes=dict(emptybarb=0.18, spacing=0.13, height=0.35),
              barb_increments=dict(half=2.5, full=5, flag=25))
    for mask, color, alpha in [(spd_s < 5,                   "#3D6B8D", 0.65),
                                ((spd_s >= 5) & (spd_s < 15),"#10933C", 0.75),
                                (spd_s >= 15,                 "#870000", 0.90)]:
        if mask.any():
            ax.barbs(lo_s[mask], la_s[mask], u_s[mask], v_s[mask],
                     color=color, alpha=alpha, **kw)


def plot_frame(data_2d, lons, lats, vmin, vmax, cfg, region_name,
               step_label, out_path, wind_u=None, wind_v=None, msl_data=None):
    extent_data = _data_extent(lons, lats)
    region_ext  = REGIONS[region_name]['extent']
    region_disp = REGIONS[region_name]['name']
    data_max, data_min = _region_stats(data_2d, lons, lats, region_ext)

    fig = plt.figure(figsize=(16, 9), facecolor='#FFFFFF', dpi=100)
    fig.text(0.030, 0.985, f"{cfg['long_name']}  ·  {region_disp}",
             fontsize=14, fontweight='bold', color='#111111', va='top', ha='left')
    fig.text(0.030, 0.96, step_label,
             fontsize=14, fontweight='bold', color='#555555', va='top', ha='left')
    fig.text(0.98, 0.985, "Made By EGTY // Model: EGTY V4",
             fontsize=14, fontweight='bold', color='#111111', va='top', ha='right')
    fig.text(0.98, 0.96,
             f"Max: {data_max:.2f}  Min: {data_min:.2f}  [{cfg['unit']}]",
             fontsize=14, fontweight='bold', color='#555555', va='top', ha='right')
    fig.add_artist(matplotlib.lines.Line2D(
        [0.03, 0.98], [0.932, 0.932], transform=fig.transFigure,
        color="black", lw=1.5, zorder=10))

    proj = ccrs.PlateCarree()
    ax   = fig.add_axes([0.03, 0.05, 0.95, 0.9], projection=proj)
    ax.set_facecolor('#E8F0E4')
    for sp in ax.spines.values():
        sp.set_edgecolor("black"); sp.set_linewidth(2)

    ax.add_feature(_OCEAN, zorder=0); ax.add_feature(_LAND, zorder=1)

    im = ax.imshow(data_2d, extent=extent_data, transform=proj, origin='upper',
                   cmap=plt.get_cmap(cfg['cmap'], 60), vmin=vmin, vmax=vmax,
                   interpolation='bilinear', zorder=3, alpha=0.82)

    if cfg['contour'] and isinstance(cfg['n_levels'], int) and cfg['n_levels'] > 0:
        lons_m, lats_m = np.meshgrid(lons, lats)
        levels = np.linspace(vmin, vmax, cfg['n_levels'] + 1)
        try:
            cs   = ax.contour(lons_m, lats_m, data_2d, levels=levels,
                              colors='#1A1A2E', linewidths=0.5, alpha=0.55,
                              transform=proj, zorder=7)
            lbls = ax.clabel(cs, inline=True, fontsize=5.5, fmt='%.0f', use_clabeltext=True)
            for lbl in lbls:
                lbl.set_color('#222222'); lbl.set_fontsize(5.5)
                lbl.set_path_effects([pe.withStroke(linewidth=1.8, foreground='white')])
        except Exception:
            pass

    ax.add_feature(_COAST, zorder=6); ax.add_feature(_BORDERS, zorder=5)
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color='#C5CDD5',
                      alpha=0.9, linestyle=':', zorder=4, x_inline=False, y_inline=False)
    gl.top_labels = False; gl.right_labels = False
    gl.xlabel_style = {'size': 8, 'color': '#555555'}
    gl.ylabel_style = {'size': 8, 'color': '#555555'}
    gl.xlocator = mticker.MultipleLocator(30); gl.ylocator = mticker.MultipleLocator(20)

    if region_ext is not None:
        ax.set_extent(region_ext, crs=proj)
    if msl_data is not None:
        _overlay_msl_isobars(ax, lons, lats, msl_data, region_ext, proj)
    if wind_u is not None and wind_v is not None:
        _overlay_wind_barbs(ax, lons, lats, wind_u, wind_v, proj)

    n_ticks = min(cfg['n_levels'] + 1, 16) if isinstance(cfg['n_levels'], int) and cfg['n_levels'] > 0 else 8
    ax_cb   = fig.add_axes([0.03, 0.03, 0.95, 0.025])
    cb = fig.colorbar(im, cax=ax_cb, orientation='horizontal',
                      ticks=np.linspace(vmin, vmax, n_ticks))
    cb.outline.set_linewidth(2); cb.outline.set_edgecolor("black")
    cb.ax.tick_params(labelsize=7.5, length=3, width=0.6,
                      color='#555555', labelcolor='#555555', direction='out')
    cb.ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    cb.ax.set_title(f"[{cfg['unit']}]", fontsize=7.5, color='#555555', pad=6, loc='right')

    plt.savefig(out_path, dpi=100, bbox_inches=None, facecolor='#FFFFFF')
    plt.close(fig)

# =============================================================================
# 9. 批次可視化
# =============================================================================

def build_all_labels(init_time_str, init_seq_len, pred_steps, step_hours=6):
    try:
        t0 = datetime.strptime(init_time_str[:13], "%Y-%m-%dT%H")
    except ValueError:
        t0 = datetime.strptime(init_time_str[:10], "%Y-%m-%d")
    labels = []
    for i in range(init_seq_len - 1, -1, -1):
        t      = t0 - timedelta(hours=i * step_hours)
        offset = -(i * step_hours)
        labels.append(f"Initial: {t.strftime('%Y-%m-%d  %H:00 UTC')}   ({offset:+d}h)")
    for i in range(1, pred_steps + 1):
        t = t0 + timedelta(hours=i * step_hours)
        labels.append(f"Forecast: {t.strftime('%Y-%m-%d  %H:00 UTC')}   (+{i*step_hours}h)")
    return labels


def compute_wind_speed(data_phys):
    u = data_phys[:, VAR_NAMES.index('u10')]
    v = data_phys[:, VAR_NAMES.index('v10')]
    return np.sqrt(u*u + v*v)


def visualize_variable(data_phys_all, var_name, lons, lats, step_labels, out_dir):
    cfg = VAR_PLOT.get(var_name)
    if cfg is None:
        print(f"[跳過] {var_name} 無繪圖設定。")
        return

    frames = (compute_wind_speed(data_phys_all)
              if var_name == 'wind_speed'
              else data_phys_all[:, VAR_NAMES.index(var_name)])

    vmin, vmax = cfg['vmin'], cfg['vmax']
    need_msl   = var_name in ('wind_speed', 'q_500')
    need_barbs = var_name == 'wind_speed'
    msl_all    = data_phys_all[:, VAR_NAMES.index('msl')] if need_msl   else None
    u10_all    = data_phys_all[:, VAR_NAMES.index('u10')] if need_barbs else None
    v10_all    = data_phys_all[:, VAR_NAMES.index('v10')] if need_barbs else None

    print(f"\n[{var_name}]  vmin={vmin:.2f}  vmax={vmax:.2f}  frames={len(step_labels)}")
    for reg_key in REGIONS:
        reg_dir     = os.path.join(out_dir, var_name, reg_key)
        os.makedirs(reg_dir, exist_ok=True)
        frame_paths = []
        for step_i, (data_2d, label) in enumerate(zip(frames, step_labels)):
            out_path = os.path.join(reg_dir, f"frame_{(6*step_i)-6:03d}.png")
            plot_frame(
                data_2d=data_2d, lons=lons, lats=lats,
                vmin=vmin, vmax=vmax, cfg=cfg,
                region_name=reg_key, step_label=label, out_path=out_path,
                wind_u=u10_all[step_i] if u10_all is not None else None,
                wind_v=v10_all[step_i] if v10_all is not None else None,
                msl_data=msl_all[step_i] if msl_all is not None else None,
            )
            frame_paths.append(out_path)

        if GIF:
            gif_path = os.path.join(reg_dir, "animation.gif")
            try:
                import imageio.v2 as imageio
                imageio.mimsave(gif_path,
                                [imageio.imread(p) for p in frame_paths],
                                duration=GIF_DURATION, loop=0)
                print(f"  [{var_name}/{reg_key}] GIF → {gif_path}")
            except ImportError:
                print("  [警告] imageio 未安裝，跳過 GIF。")
            except Exception as e:
                print(f"  [GIF 警告] {e}")

# =============================================================================
# 10. 主流程
# =============================================================================

def _load_model(n_weather, n_static):
    if not os.path.exists(MODEL_PATH):
        print(f"下載模型 {MODEL_URL} ...")
        resp = requests.get(MODEL_URL)
        resp.raise_for_status()
        with open(MODEL_PATH, 'wb') as f:
            f.write(resp.content)
    model = CircularUNet3D(n_weather + n_static, BASE_CH,
                           n_weather, GROUP_NORM_GROUPS).to(DEVICE)
    ckpt  = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"模型已載入: {MODEL_PATH}")
    return model


def _manage_forecast_dirs(tag):
    existing = sorted(
        [d for d in os.listdir(".") if os.path.isdir(d) and d.startswith("forecast_")],
        reverse=True)
    if tag not in existing:
        existing.insert(0, tag)
    while len(existing) > 3:
        oldest = existing.pop()
        try:
            shutil.rmtree(oldest)
            print(f"已刪除最舊資料夾: {oldest}")
        except Exception as e:
            print(f"刪除 {oldest} 失敗: {e}")
    forecast_data = [{"folder": d.replace("forecast_", "")} for d in existing]
    with open("forecast_list.json", "w", encoding="utf-8") as f:
        json.dump({"EGTY_V4": forecast_data}, f, indent=4, ensure_ascii=False)
    print(f"清單已更新：{len(forecast_data)} 個預報時段。")


def main():
    # ── 載入靜態場與初始資料 ─────────────────────────────────────────────
    static_tensor = load_static(STATIC_PATHS)

    if not os.path.exists(TEST_NPZ_PATH):
        raise FileNotFoundError(f"找不到輸入資料: {TEST_NPZ_PATH}")
    npz      = np.load(TEST_NPZ_PATH, allow_pickle=True)
    data_std = npz['data']
    times    = npz['times']
    print(f"資料時間: {times}")

    T, C, H, W = data_std.shape
    lats = np.linspace(90, -90 + 0.5, H)
    lons = np.linspace(-180, 180 - 0.5, W)

    if T < SEQ_LEN:
        raise ValueError(f"資料長度 {T} < SEQ_LEN {SEQ_LEN}")
    init_seq = data_std[:SEQ_LEN]

    # ── 目錄管理 ──────────────────────────────────────────────────────────
    last_time_str = str(times[-1])
    tag     = f"forecast_{last_time_str.replace(':', '').replace(' ', '_')}"
    OUT_DIR = f"./{tag}/V4_model"
    _manage_forecast_dirs(tag)

    if os.path.exists(OUT_DIR):
        print(f"目錄 {OUT_DIR} 已存在，略過計算。")
        return False

    # ── 載入模型 ──────────────────────────────────────────────────────────
    n_static = static_tensor.shape[0] if static_tensor is not None else 0
    model    = _load_model(len(VAR_NAMES), n_static)

    means = [FIXED_STATS[v]['mean'] for v in VAR_NAMES]
    stds  = [FIXED_STATS[v]['std']  for v in VAR_NAMES]

    # ── 第一次預測：原始（deterministic） ─────────────────────────────────
    print("\n=== [1/6] 原始預測 ===")
    preds_std_det = autoregressive_predict(
        model, init_seq, static_tensor, AUTOREGRESSIVE_STEPS, DEVICE,
        desc="原始預測")
    init_phys_det  = destandardize(init_seq,      means, stds)
    preds_phys_det = destandardize(preds_std_det, means, stds)

    # ── 連續7次微擾預測 ──────────────────────────────────────────────────
    all_preds_phys = [preds_phys_det]
    all_init_phys  = [init_phys_det]
    for sample_idx in range(7):
        print(f"\n=== [{sample_idx + 2}/6] 微擾預測 (樣本 {sample_idx + 1}/7) ===")
        init_seq_pert = perturb_init_seq(init_seq, seed=SEED + sample_idx + 1)
        preds_std_pert = autoregressive_predict(
            model, init_seq_pert, static_tensor,
            PERTURBED_AUTOREGRESSIVE_STEPS, DEVICE,
            desc=f"微擾預測 #{sample_idx + 1}")
        init_phys_pert  = destandardize(init_seq_pert,  means, stds)
        preds_phys_pert = destandardize(preds_std_pert, means, stds)
        all_preds_phys.append(preds_phys_pert)
        all_init_phys.append(init_phys_pert)

    # ── 颱風追蹤（原始 + 7個微擾樣本） ───────────────────────────────────
    active_list = get_active_typhoon_initials("active_typhoon")
    base_time   = datetime.strptime(last_time_str, "%Y-%m-%dT%H")

    if not active_list:
        print("[資訊] 未發現活躍颱風，略過追蹤。")
    else:
        all_tracks = {}
        for ty in active_list:
            print(f"\n--- 追蹤颱風 {ty['name']} ({ty['id']}) ---")
            tracks_list = []

            # 追蹤所有樣本（原始 + 7個微擾）
            for sample_id, (init_phys, preds_phys) in enumerate(zip(all_init_phys, all_preds_phys)):
                init_point = _build_init_point(init_phys[-1], ty, base_time)
                desc_prefix = "DET" if sample_id == 0 else f"PERT#{sample_id}"
                
                predicted = track_typhoon(
                    preds_phys, [init_point],
                    ty['id'], base_time, desc_prefix=desc_prefix)
                
                tracks_list.append({
                    "sample_id": float(sample_id),
                    "data_points": [init_point] + predicted,
                })

            all_tracks[ty['id']] = tracks_list

        output_json = os.path.join("active_typhoon", "cyclone_data_egty4.json")
        os.makedirs("active_typhoon", exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(convert_numpy(all_tracks), f, indent=4, ensure_ascii=False)
        print(f"\n✅ 颱風路徑（原始 + 7個微擾樣本）已儲存至: {output_json}")

    # ── 可視化（僅用原始預測） ────────────────────────────────────────────
    print("\n=== 可視化（原始預測） ===")
    all_phys    = np.concatenate([all_init_phys[0], all_preds_phys[0]], axis=0)
    all_phys    = np.roll(all_phys, W // 2, axis=-1)
    step_labels = build_all_labels(last_time_str, SEQ_LEN, AUTOREGRESSIVE_STEPS)

    os.makedirs(OUT_DIR, exist_ok=True)
    for vname in VIS_VARS:
        visualize_variable(all_phys, vname, lons, lats, step_labels, OUT_DIR)

    print(f"\n✅ 所有結果已儲存至: {OUT_DIR}")


if __name__ == "__main__":
    main()
