import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime
from shapely.validation import make_valid
import os
import shutil
import glob
from pyproj import Geod
from shapely.geometry import Polygon
from shapely.ops import unary_union
import cartopy.crs as ccrs

# ── Saffir-Simpson 色彩分級 ────────────────────────────────────────────────────
CATEGORIES = [
    {"name": "Category 5",     "min_kt": 137, "color": "#F700FF"},
    {"name": "Category 4",     "min_kt": 113, "color": "#C90000"},
    {"name": "Category 3",     "min_kt":  96, "color": "#FF5100"},
    {"name": "Category 2",     "min_kt":  83, "color": "#D29E00"},
    {"name": "Category 1",     "min_kt":  64, "color": "#CAC600"},
    {"name": "Tropical Storm", "min_kt":  34, "color": "#00A424"},
    {"name": "Depression",     "min_kt":  25, "color": "#0047A4"},
    {"name": "Disturbance",    "min_kt":   0, "color": "#7C7C7C"},
]

# ── 模型設定：新增模型在此擴充即可 ─────────────────────────────────────────────
MODEL_CONFIGS = {
    "FNV3": {
        "json_path":   "active_typhoon/cyclone_data_fnv3.json",
},
    "GENC": {
        "json_path":   "active_typhoon/cyclone_data_genc.json",
    },
    "EGTY_V4": {
        "json_path":   "active_typhoon/cyclone_data_egty4.json",
    },
}
ERROR_RADII = { #誤差圈半徑（公里）對應預報小時數
    0:   0,
    24:  100,
    48:  280,
    72:  500,
    96:  800,
    120: 1000,
    144: 1500,
    168: 2000,
    180: 2500,
}

# ══════════════════════════════════════════════════════════════════════════════
# 工具函式
# ══════════════════════════════════════════════════════════════════════════════

def get_error_radius(hours: float) -> float:
    # 簡單線性插值或取最接近值
    times = sorted(ERROR_RADII.keys())
    return np.interp(hours, times, [ERROR_RADII[t] for t in times])

def get_color(wind_kt: float) -> str:
    for cat in CATEGORIES:
        if wind_kt >= cat["min_kt"]:
            return cat["color"]
    return CATEGORIES[-1]["color"]

def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

_FMTS = (
    "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M",    "%Y-%m-%dT%H", "%Y%m%d%H",
)

def parse_dt(raw: str):
    for fmt in _FMTS:
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            pass
    return None

def unwrap_lons(lons) -> np.ndarray:
    return np.degrees(np.unwrap(np.radians(np.array(lons, dtype=float))))

# ══════════════════════════════════════════════════════════════════════════════
# 地圖範圍計算
# ══════════════════════════════════════════════════════════════════════════════

def compute_map_params(tracks: list, pad_deg: float = 5.0):
    all_lons = np.concatenate([t["lons"] for t in tracks])
    all_lats = np.concatenate([t["lats"] for t in tracks])

    lon_min_uw, lon_max_uw = all_lons.min(), all_lons.max()
    lat_min,    lat_max    = all_lats.min(), all_lats.max()

    lon_center = (lon_min_uw + lon_max_uw) / 2.0
    lat_center = (lat_min    + lat_max)    / 2.0

    raw_half_lon = (lon_max_uw - lon_min_uw) / 2.0 + pad_deg
    raw_half_lat = (lat_max    - lat_min)    / 2.0 + pad_deg

    target_aspect  = 4.0 / 3.0
    current_aspect = raw_half_lon / raw_half_lat

    if current_aspect > target_aspect:
        half_lon = raw_half_lon
        half_lat = raw_half_lon / target_aspect
    else:
        half_lat = raw_half_lat
        half_lon = raw_half_lat * target_aspect

    central_lon = ((lon_center + 180) % 360) - 180
    extent_proj = (
        lon_center - half_lon,
        lon_center + half_lon,
        lat_center - half_lat,
        lat_center + half_lat,
    )
    return central_lon, extent_proj, lon_center, lat_center

# ══════════════════════════════════════════════════════════════════════════════
# JSON 結構解析
# ══════════════════════════════════════════════════════════════════════════════

def parse_tracks(samples: list) -> list:
    tracks = []
    for sample in samples:
        pts = sample["data_points"]
        if not pts:
            continue
        times, lats, lons, winds, pressure = [], [], [], [], []
        for p in pts:
            times.append(p["valid_time"])
            lats.append(p["coordinates"]["lat"])
            lons.append(p["coordinates"]["lon"])
            winds.append(p["intensity"]["max_wind_knots"])
            pressure.append(p["intensity"]["mslp_hpa"])

        tracks.append({
            "sample_id":    sample["sample_id"],
            "times":        times,
            "lats":         np.array(lats,     dtype=float),
            "lons":         unwrap_lons(lons),
            "winds":        np.array(winds,    dtype=float),
            "pressure":     np.array(pressure, dtype=float),
            "max_wind":     float(np.nanmax(winds)),
            "min_pressure": float(np.nanmin(pressure)),
        })
    return tracks

def ensemble_mean(tracks: list):
    if not tracks:
        return None
    n = min(len(t["lats"]) for t in tracks)
    return {
        "lats":         np.nanmean([t["lats"][:n]  for t in tracks], axis=0),
        "lons":         np.nanmean([t["lons"][:n]  for t in tracks], axis=0),
        "winds":        np.nanmean([t["winds"][:n] for t in tracks], axis=0),
        "times":        tracks[0]["times"][:n],
        "max_wind":     float(np.nanmean([t["max_wind"]     for t in tracks])),
        "min_pressure": float(np.nanmean([t["min_pressure"] for t in tracks])),
    }

# ══════════════════════════════════════════════════════════════════════════════
# 繪圖輔助
# ══════════════════════════════════════════════════════════════════════════════

def get_geodesic_circle(lon, lat, radius_km, num_points=60):
    geod = Geod(ellps='WGS84')
    angles = np.linspace(0, 360, num_points)
    lons, lats, _ = geod.fwd([lon]*num_points, [lat]*num_points, angles, [radius_km * 1000]*num_points)
    
    # 關鍵修正：處理跨越換日線的連續性
    # 確保所有點相對於圓心 lon 的偏移不超過 180 度
    lons = np.array(lons)
    lons = ((lons - lon + 180) % 360) + lon - 180
    
    return Polygon(zip(lons, lats))

def draw_forecast_cone(ax, track, data_crs):
    """
    針對單一軌跡繪製預報扇形
    """
    lons = track["lons"]
    lats = track["lats"]
    times = [parse_dt(t) for t in track["times"]]
    t0 = times[0]
    
    circles = []
    for i, (lon, lat, t) in enumerate(zip(lons, lats, times)):
        if t is None: continue
        hours = (t - t0).total_seconds() / 3600
        radius = get_error_radius(hours)
        circles.append(get_geodesic_circle(lon, lat, radius))
    
    segments = []
    for i in range(len(circles) - 1):
        # 建立兩圓之間的凸包
        segment = circles[i].union(circles[i+1]).convex_hull
        segments.append(segment)
    
    full_cone = unary_union(segments)
    
    # 繪製到地圖上
    if not full_cone.is_empty:
        # 轉換為外框線與填充
        ax.add_geometries(
            [full_cone], 
            crs=ccrs.PlateCarree(), 
            facecolor="#F3FF4A",  # 黃色填充
            edgecolor='#cc3300',  # 橘色外框
            alpha=0.4,            # 透明度
            zorder=2
        )

def draw_track(ax, lons, lats, winds, lw=0.9, alpha=0.70, zorder=2, data_crs=None):
    if data_crs is None:
        data_crs = ccrs.PlateCarree()
    for i in range(len(lons) - 1):
        c = get_color((winds[i] + winds[i + 1]) / 2)
        ax.plot(
            [lons[i], lons[i + 1]], [lats[i], lats[i + 1]],
            color=c, linewidth=lw, alpha=alpha,
            transform=data_crs, zorder=zorder, solid_capstyle="round",
        )

def draw_mean_track(ax, mean: dict, data_crs, interval_h: int = 24):
    lons, lats = mean["lons"], mean["lats"]
    ax.plot(lons[0], lats[0], "kx", markersize=11, markeredgewidth=2.5,
            transform=data_crs, zorder=7)
    parsed = [parse_dt(r) for r in mean["times"]]
    t0, done = parsed[0], set()
    for i, t in enumerate(parsed):
        if t is None or t0 is None:
            continue
        elh = (t - t0).total_seconds() / 3600
        lh  = round(elh / interval_h) * interval_h
        if lh > 0 and lh not in done:
            done.add(lh)
            ax.plot(lons[i], lats[i], "ks", markersize=7,
                    transform=data_crs, zorder=7)
            ax.text(
                lons[i] + 0.7, lats[i], f"+{int(lh)}h",
                fontsize=8.5, fontweight="bold", color="black",
                transform=data_crs, zorder=8,
                path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
            )

def draw_member_list(fig, tracks: list, mean: dict):
    items = sorted(tracks, key=lambda t: (t["max_wind"], -t["min_pressure"]), reverse=True)
    ax2   = fig.add_axes([0.843, 0.12, 0.15, 0.8])
    ax2.axis("off")
    for i in range(len(items) + 1):
        y = 1.0 - i / 47
        if i < len(items):
            t     = items[i]
            color = get_color(t["max_wind"])
            ax2.text(0.05, y, f'#{int(t["sample_id"] + 1)}',
                     fontsize=10, color=color, va="center", ha="left", fontfamily="monospace")
            ax2.text(0.25, y,f'{t["max_wind"]:.1f}',
                     fontsize=10, color=color, va="center", ha="left", fontfamily="monospace")
            ax2.text(0.45, y,
                     f'kt  {mean["min_pressure"]:.1f}hPa' if mean["min_pressure"] < 1000 else f'{int(mean["min_pressure"])}hPa',
                     fontsize=10, color=color, va="center", ha="left", fontfamily="monospace")
        else:
            color = get_color(mean["max_wind"])
            ax2.text(0.00, y - 0.03, "mean",
                     fontsize=10, color="black", va="center", ha="left", fontfamily="monospace")
            ax2.text(0.25, y - 0.03,f'{mean["max_wind"]:.1f}',
                     fontsize=10, color=color, va="center", ha="left", fontfamily="monospace")
            ax2.text(0.45, y - 0.03,
                    f'kt  {mean["min_pressure"]:.1f}hPa' if mean["min_pressure"] < 1000 else f'{int(mean["min_pressure"])}hPa',
                     fontsize=10, color=color, va="center", ha="left", fontfamily="monospace")

# ══════════════════════════════════════════════════════════════════════════════
# 主繪圖（支援多模型）
# ══════════════════════════════════════════════════════════════════════════════

def plot_one_track(track_id: str, samples: list, output_dir: str, model_name: str = "MODEL"):
    tracks = parse_tracks(samples)
    if not tracks:
        print(f"  [SKIP] {track_id}：無有效軌跡資料")
        return

    mean = ensemble_mean(tracks)
    central_lon, extent_proj, _, _ = compute_map_params(tracks)

    proj     = ccrs.PlateCarree(central_longitude=central_lon)
    data_crs = ccrs.PlateCarree()
    lon_min_e, lon_max_e, lat_min_e, lat_max_e = extent_proj

    def to_standard(lon_uw):
        return ((lon_uw - central_lon + 180) % 360) - 180 + central_lon

    fig = plt.figure(figsize=(13.5, 9), facecolor="white")
    ax  = fig.add_axes([0.01, 0.04, 0.85, 0.89], projection=proj)
    ax.set_extent(
        [to_standard(lon_min_e), to_standard(lon_max_e), lat_min_e, lat_max_e],
        crs=data_crs,
    )

    fig.add_artist(matplotlib.lines.Line2D(
        [0.039, 0.99], [0.937, 0.937],
        transform=fig.transFigure, color="black", lw=1, zorder=10,
    ))

    ax.add_feature(cfeature.LAND,      facecolor="#C8C8C8", edgecolor="#888888", linewidth=0.5, zorder=1)
    ax.add_feature(cfeature.OCEAN,     facecolor="#E8F4FA", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="#777777", zorder=1)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.4, edgecolor="#AAAAAA", zorder=1)
    ax.add_feature(cfeature.LAKES,     facecolor="#E8F4FA", edgecolor="#AAAAAA", linewidth=0.3, zorder=1)

    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="gray", alpha=0.5,
                      linestyle="--", crs=data_crs)
    gl.top_labels   = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 8}
    gl.ylabel_style = {"size": 8}

    if model_name == "EGTY_V4" or len(tracks) == 1:
        draw_forecast_cone(ax, tracks[0], data_crs)
        # 依然畫出主路徑線條，但可以加粗
        draw_track(ax, tracks[0]["lons"], tracks[0]["lats"], tracks[0]["winds"], 
                   lw=2.5, alpha=1.0, zorder=3, data_crs=data_crs)
    else:
        for tr in tracks:
            draw_track(ax, tr["lons"], tr["lats"], tr["winds"],
                       lw=0.9, alpha=0.65, zorder=2, data_crs=data_crs)
            
    if mean:
        draw_mean_track(ax, mean, data_crs)

    handles = [mpatches.Patch(facecolor=c["color"], label=c["name"]) for c in CATEGORIES]
    ax.legend(handles=handles, loc="lower left", fontsize=8,
              framealpha=0.88, edgecolor="#AAAAAA", facecolor="white")

    init_dt  = parse_dt(tracks[0]["times"][0])
    init_str = init_dt.strftime("%Y-%m-%dT%H")

    ax.annotate(
        f"{model_name} Ensemble Forecast [{track_id}] Tropical Cyclone Track — {len(tracks)} Members\n"
        f"Maximum 1-minute Sustained Wind Speed and Minimum Central Pressure",
        xy=(0, 1.01), xycoords="axes fraction", textcoords="offset points",
        ha="left", va="bottom", fontsize=13, color="#444444", fontweight="bold",
    )
    ax.annotate(
        f"Made By EGTY\n{init_str}:Forecast time",
        xy=(1.2, 1.01), xycoords="axes fraction", textcoords="offset points",
        ha="right", va="bottom", fontsize=13, color="#444444", fontweight="bold",
    )

    draw_member_list(fig, tracks, mean)

    safe_id  = track_id.replace("/", "_").replace("\\", "_")
    out_path = os.path.join(output_dir, safe_id, model_name, f"{init_str}.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=100, bbox_inches=None, facecolor="white")
    plt.close(fig)
    print(f"  ✅  [{model_name}] {track_id}  →  {out_path}  (central_lon={central_lon:.1f}°)")

# ══════════════════════════════════════════════════════════════════════════════
# 清理舊圖
# ══════════════════════════════════════════════════════════════════════════════

def cleanup_old_tracks(output_dir: str):
    if not os.path.exists(output_dir):
        return

    now = datetime.utcnow()
    for track_id in os.listdir(output_dir):
        track_path = os.path.join(output_dir, track_id)
        if not os.path.isdir(track_path):
            continue

        # 逐模型子資料夾檢查
        for model_name in os.listdir(track_path):
            model_dir = os.path.join(track_path, model_name)
            if not os.path.isdir(model_dir):
                continue

            png_files = sorted(glob.glob(os.path.join(model_dir, "*.png")))
            if not png_files:
                continue

            # 規則 1：超過 4 張刪最舊
            if len(png_files) > 4:
                for f in png_files[:-4]:
                    try:
                        os.remove(f)
                        print(f"🗑️ [{model_name}] 刪除舊圖 (超過4張): {f}")
                    except Exception as e:
                        print(f"⚠️ 刪除失敗 {f}: {e}")
                png_files = png_files[-4:]

            # 規則 2：最新圖超過 24 小時刪整個風暴資料夾
            newest_name = os.path.basename(png_files[-1]).replace(".png", "")
            try:
                newest_time = datetime.strptime(newest_name, "%Y-%m-%dT%H")
                if (now - newest_time).total_seconds() > 24 * 3600:
                    shutil.rmtree(track_path)
                    print(f"🧹 {track_id} 超過 24h 未更新，已刪除: {track_path}")
                    break   # 整個風暴資料夾已刪，跳出模型迴圈
            except ValueError:
                print(f"⚠️ 無法從檔名解析時間: {newest_name}")

# ══════════════════════════════════════════════════════════════════════════════
# 單一模型的完整流程
# ══════════════════════════════════════════════════════════════════════════════

def run_model_pipeline(model_name: str, output_dir: str):
    cfg = MODEL_CONFIGS[model_name]
    print(f"\n{'='*60}")
    print(f"  模型：{model_name}")
    print(f"{'='*60}")

    # 3. 繪圖
    if not os.path.exists(cfg["json_path"]):
        print(f"❌ [{model_name}] 找不到 {cfg['json_path']}，跳過繪圖。")
        return

    data = load_json(cfg["json_path"])
    print(f"🌀 [{model_name}] 共 {len(data)} 個 track_id，開始繪圖…")
    for track_id, samples in data.items():
        plot_one_track(track_id, samples, output_dir, model_name=model_name)
    print(f"🎉 [{model_name}] 繪圖完成")

# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════

def main():
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "active_typhoon")
    os.makedirs(output_dir, exist_ok=True)

    # 依序跑每個模型
    for model_name in MODEL_CONFIGS:
        run_model_pipeline(model_name, output_dir)

    # 統一清理
    print(f"\n{'='*60}")
    print("🧹 開始自動清理…")
    cleanup_old_tracks(output_dir)
    print("✅ 清理完成！")

if __name__ == "__main__":
    main()
