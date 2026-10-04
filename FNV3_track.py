import json
import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime
from matplotlib.collections import LineCollection
import os
import shutil
import glob
from pyproj import Geod
from shapely.geometry import Polygon
from shapely.ops import unary_union
import warnings

warnings.filterwarnings(action="ignore", message="Mean of empty slice")

# ── Saffir-Simpson 色彩分級 ──────────────────────────────────────────────────
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

# ── 模型設定 ─────────────────────────────────────────────────────────────────
MODEL_CONFIGS = {
    "FNV3":    {"json_path": "active_typhoon/cyclone_data_fnv3.json"},
    "GFSE":    {"json_path": "active_typhoon/cyclone_data_gfse.json"},
    "GENC":    {"json_path": "active_typhoon/cyclone_data_genc.json"},
    "EGTY_V4": {"json_path": "active_typhoon/cyclone_data_egty4.json"},
}

ERROR_RADII = {
    0: 50, 24: 100, 48: 150, 72: 200, 96: 300, 120: 500, 192: 700,
}

# ── 4模型比較圖專用線色（固定，與 Saffir-Simpson 無關）────────────────────────
MODEL_COLORS = {
    "FNV3":    "#E63946",   # 紅
    "GFSE":    "#2196F3",   # 藍
    "GENC":    "#FF9800",   # 橙
    "EGTY_V4": "#4CAF50",   # 綠
}

_GEOD = Geod(ellps="WGS84")

_MAP_FEATURES = [
    (cfeature.LAND,      dict(facecolor="#C8C8C8", edgecolor="#888888", linewidth=0.5, zorder=1)),
    (cfeature.OCEAN,     dict(facecolor="#E8F4FA", zorder=0)),
    (cfeature.COASTLINE, dict(linewidth=0.6, edgecolor="#777777", zorder=1)),
    (cfeature.BORDERS,   dict(linewidth=0.4, edgecolor="#AAAAAA", zorder=1)),
    (cfeature.LAKES,     dict(facecolor="#E8F4FA", edgecolor="#AAAAAA", linewidth=0.3, zorder=1)),
]

_ERROR_RADII_TIMES  = np.array(sorted(ERROR_RADII.keys()), dtype=float)
_ERROR_RADII_VALUES = np.array([ERROR_RADII[t] for t in _ERROR_RADII_TIMES], dtype=float)

SECTOR_BEARINGS = [(0, 90), (90, 180), (180, 270), (270, 360)]

# ── 台灣範圍定義 ─────────────────────────────────────────────────────────────
TAIWAN_INSET_EXTENT = (119.0, 123.0, 21.0, 26.0)   # (lon_min, lon_max, lat_min, lat_max)
TAIWAN_INSET_RECT   = [0.025, 0.62, 0.20, 0.305]  # [left, bottom, width, height]


# ══════════════════════════════════════════════════════════════════════════════
# 基礎工具
# ══════════════════════════════════════════════════════════════════════════════

def get_error_radius(hours: float) -> float:
    return float(np.interp(hours, _ERROR_RADII_TIMES, _ERROR_RADII_VALUES))


def get_color(wind_kt: float) -> str:
    for cat in CATEGORIES:
        if wind_kt >= cat["min_kt"]:
            return cat["color"]
    return CATEGORIES[-1]["color"]


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def parse_dt(raw: str) -> datetime:
    return datetime.strptime(raw, "%Y-%m-%d %H:%M:%S")


def _safe_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_pressure_text(pressure):
    return f" / {pressure:.1f} hPa" if pressure is not None else ""


def _format_current_intensity(current_wind, current_pressure):
    return f"Current intensity: {current_wind:.1f} kt{_format_pressure_text(current_pressure)}"


def _build_plot_title(model_name: str, track_id: str, current_wind: float,
                      current_pressure, is_cone_model: bool) -> str:
    subtitle = _format_current_intensity(current_wind, current_pressure)
    if is_cone_model:
        return f"{model_name} [{track_id}] Path and Storm Radius Forecast Plot\n{subtitle}"
    return f"{model_name} [{track_id}] ensemble plot\n{subtitle}"


def unwrap_lons(lons) -> np.ndarray:
    return np.degrees(np.unwrap(np.radians(np.array(lons, dtype=float))))


def align_lon_to_ref(lon_array, ref_lon: float, anchor: str = "first") -> np.ndarray:
    """
    將整條經度序列整體平移 360° 的整數倍，使其與 ref_lon 落在同一個換日線分支上。

    背景：unwrap_lons() 只能保證單一條序列內部連續，但無法保證多條序列彼此落在
    同一個分支（例如一條展開成 179→181，另一條展開成 -181→-179，兩者物理上相鄰，
    數值卻差了 360°）。任何要「跨序列」做平均、比較、銜接的地方，都必須先用本函式
    對齊，否則會出現平均路徑瞬間跳到地球另一端，或拼接出橫跨整張地圖的離譜線段。
    """
    lon_array = np.asarray(lon_array, dtype=float)
    if len(lon_array) == 0:
        return lon_array
    anchor_val = lon_array[0] if anchor == "first" else lon_array[-1]
    shift = np.round((ref_lon - anchor_val) / 360.0) * 360.0
    return lon_array + shift if shift != 0 else lon_array


def align_tracks_lon(tracks: list, ref_lon: float | None = None) -> list:
    """
    讓同一批 tracks（例如同一個 ensemble 裡的所有 member，或多模型平均路徑）
    全部對齊到同一個換日線分支，避免互相平均 / 比較時數值相差 360°。
    """
    if not tracks:
        return tracks
    if ref_lon is None:
        ref_lon = tracks[0]["lons"][0]
    for t in tracks:
        t["lons"] = align_lon_to_ref(t["lons"], ref_lon, anchor="first")
    return tracks


def normalize_radius_to_quadrants(radius):
    if isinstance(radius, dict):
        return [
            float(radius.get(key, np.nan)) if radius.get(key) is not None else np.nan
            for key in ("NE", "SE", "SW", "NW")
        ]
    if isinstance(radius, (list, tuple, np.ndarray)):
        quad = [float(v) if v is not None else np.nan for v in list(radius)[:4]]
    elif radius is None or (isinstance(radius, float) and np.isnan(radius)):
        quad = [np.nan] * 4
    else:
        quad = [float(radius)] * 4
    while len(quad) < 4:
        quad.append(np.nan)
    return quad


def is_in_taiwan_range(tracks: list) -> bool:
    """
    檢查傳入的所有 track 中是否有任何一個軌跡點落在台灣指定範圍內。
    """
    lon_min, lon_max, lat_min, lat_max = TAIWAN_INSET_EXTENT
    for tr in tracks:
        lons = tr["lons"]
        lats = tr["lats"]
        lons_norm = ((lons + 180) % 360) - 180
        in_lon = (lons_norm >= lon_min) & (lons_norm <= lon_max)
        in_lat = (lats >= lat_min) & (lats <= lat_max)
        if np.any(in_lon & in_lat):
            return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
# 平均路徑計算（JSON 層，供 cone sample 使用）
# ══════════════════════════════════════════════════════════════════════════════

def _avg_lon_circular(values):
    """
    經度專用平均：先把所有值對齊到同一分支（以第一個有效值為基準）再做算術平均，
    避免像 179.5 與 -179.8 這種「物理上相鄰、數值上相差 360°」的情況被平均成 0
    （也就是平均路徑瞬間跳到地球另一端的 bug）。
    """
    values = [v for v in values if v is not None]
    if not values:
        return None
    ref = values[0]
    aligned = [v - round((v - ref) / 360.0) * 360.0 for v in values]
    return float(sum(aligned) / len(aligned))


def calculate_average_track(samples: list) -> dict | None:
    if not samples:
        return None

    merged = {}
    for sample in samples:
        if sample.get("sample_id") == -1.0:
            continue
        for point in sample["data_points"]:
            vt = point["valid_time"]
            if vt not in merged:
                merged[vt] = {
                    "lats": [], "lons": [], "mslps": [], "winds": [],
                    "radii": {"NE": [], "SE": [], "SW": [], "NW": []},
                }
            merged[vt]["lats"].append(_safe_float(point["coordinates"]["lat"]))
            merged[vt]["lons"].append(_safe_float(point["coordinates"]["lon"]))
            merged[vt]["mslps"].append(_safe_float(point["intensity"].get("mslp_hpa")))
            merged[vt]["winds"].append(_safe_float(point["intensity"].get("max_wind_knots")))
            radii = point["intensity"].get("r34_quadrant_km")
            if isinstance(radii, dict):
                for key, value in radii.items():
                    merged[vt]["radii"][key].append(_safe_float(value))

    def avg(values):
        values = [v for v in values if v is not None]
        return float(sum(values) / len(values)) if values else None

    initial_member_count = len([s for s in samples if s.get("sample_id") != -1.0])
    threshold = initial_member_count * 0.6

    data_points = []
    for vt in sorted(merged.keys(), key=lambda x: parse_dt(x) if x else x):
        entry = merged[vt]
        available_members = len([v for v in entry["lats"] if v is not None])
        if available_members < threshold:
            print(
                f"   ℹ️  平均路徑計算停止：{vt} 時間點成員數 {available_members} "
                f"< 初始成員數 {initial_member_count} 的 60%。"
            )
            break

        intensity = {
            "mslp_hpa": avg(entry["mslps"]),
            "max_wind_knots": avg(entry["winds"]),
        }
        quadrant = {k: avg(v) for k, v in entry["radii"].items()}
        if any(value is not None for value in quadrant.values()):
            intensity["r34_quadrant_km"] = quadrant

        data_points.append({
            "valid_time": vt,
            "coordinates": {"lat": avg(entry["lats"]), "lon": _avg_lon_circular(entry["lons"])},
            "intensity": intensity,
        })

    return {"sample_id": -1.0, "data_points": data_points}


# ══════════════════════════════════════════════════════════════════════════════
# JSON 結構解析
# ══════════════════════════════════════════════════════════════════════════════

def parse_tracks(samples: list) -> list:
    tracks = []
    for sample in samples:
        pts = sample["data_points"]
        if not pts:
            continue
        times, lats, lons, winds, pressure, radii = [], [], [], [], [], []
        for p in pts:
            times.append(parse_dt(p["valid_time"]))
            lats.append(p["coordinates"]["lat"])
            lons.append(p["coordinates"]["lon"])
            winds.append(p["intensity"]["max_wind_knots"])
            pressure.append(p["intensity"]["mslp_hpa"])
            radius = p["intensity"].get("r34_quadrant_km")
            if radius is None:
                radius = p["intensity"].get("avg_7_level_wind_radius_km", np.nan)
            radii.append(normalize_radius_to_quadrants(radius))

        tracks.append({
            "sample_id": sample["sample_id"],
            "times":     times,
            "lats":      np.array(lats,     dtype=float),
            "lons":      unwrap_lons(lons),
            "winds":     np.array(winds,    dtype=float),
            "pressure":  np.array(pressure, dtype=float),
            "radii":     np.array(radii,    dtype=float),
            "max_wind":      float(np.nanmax(winds)) if any(np.isfinite(winds)) else 0.0,
            "min_pressure":  float(np.nanmin(pressure)) if any(np.isfinite(pressure)) else 1010.0,
        })

    # 重要：每條 member 各自 unwrap 後，可能落在不同的換日線分支上
    # （例如一條展開成 179→181，另一條展開成 -181→-179）。
    # 這裡統一對齊到第一條的分支，確保後續所有跨 member 的運算
    # （ensemble_mean、compute_member_spread_radius、繪圖比較…）都在同一參考下進行。
    return align_tracks_lon(tracks)


def ensemble_mean(tracks: list, min_members: int = 2) -> dict | None:
    actual_members = [t for t in tracks if t["sample_id"] != -1.0]
    if not actual_members:
        return None

    initial_member_count = len(actual_members)
    threshold_60pct = initial_member_count * 0.6
    effective_threshold = max(min_members, threshold_60pct)

    max_n = max(len(t["lats"]) for t in actual_members)
    mean_lats, mean_lons, mean_winds, mean_pressure, mean_times, mean_radii = [], [], [], [], [], []

    for i in range(max_n):
        members = [t for t in actual_members if i < len(t["lats"])]
        n_valid = len(members)

        if n_valid >= effective_threshold:
            mean_lats.append(np.nanmean([t["lats"][i]    for t in members]))
            mean_lons.append(np.nanmean([t["lons"][i]    for t in members]))
            mean_winds.append(np.nanmean([t["winds"][i]  for t in members]))
            mean_pressure.append(np.nanmean([t["pressure"][i] for t in members]))
            mean_times.append(members[0]["times"][i])
            mean_radii.append(np.nanmean([t["radii"][i]  for t in members], axis=0))
        else:
            print(
                f"   ℹ️  ensemble_mean 計算停止：時間索引 {i}，"
                f"有效成員數 {n_valid} < 門檻 {effective_threshold:.1f} "
                f"（初始 {initial_member_count} × 60%）"
            )
            break

    if not mean_lats:
        return None

    return {
        "lats":         np.array(mean_lats,     dtype=float),
        "lons":         np.array(mean_lons,     dtype=float),
        "winds":        np.array(mean_winds,    dtype=float),
        "pressure":     np.array(mean_pressure, dtype=float),
        "times":        mean_times,
        "radii":         np.array(mean_radii,    dtype=float),
        "max_wind":     float(np.nanmean([t["max_wind"]     for t in actual_members])),
        "min_pressure": float(np.nanmean([t["min_pressure"] for t in actual_members])),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 過去路徑
# ══════════════════════════════════════════════════════════════════════════════

def parse_best_track_dat(dat_path: str, init_dt: datetime) -> dict | None:
    if not os.path.exists(dat_path):
        return None

    times, lats, lons, winds = [], [], [], []
    with open(dat_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 9:
                continue
            tech    = parts[4]
            tau_str = parts[5]
            if tech != "BEST" and tau_str != "0":
                continue
            try:
                dt = datetime.strptime(parts[2], "%Y%m%d%H")
            except ValueError:
                continue
            if dt > init_dt:
                continue

            lat_str, lon_str = parts[6], parts[7]
            try:
                lat_val = float(lat_str[:-1]) / 10.0
                if lat_str[-1] == "S":
                    lat_val = -lat_val
                lon_val = float(lon_str[:-1]) / 10.0
                if lon_str[-1] == "W":
                    lon_val = -lon_val
                wind_val = float(parts[8])
            except Exception:
                continue

            if times and times[-1] == dt:
                if tech == "BEST":
                    lats[-1], lons[-1], winds[-1] = lat_val, lon_val, wind_val
                continue

            times.append(dt)
            lats.append(lat_val)
            lons.append(lon_val)
            winds.append(wind_val)

    if not times:
        return None

    return {
        "times": times,
        "lats":  np.array(lats,  dtype=float),
        # 修正：原本對單一數值呼叫 unwrap_lons 等同 no-op，整條序列從未真正展開過。
        # 改成對整條 lons 序列做 unwrap，確保 past_track 本身在跨換日線時是連續的。
        "lons":  unwrap_lons(lons),
        "winds": np.array(winds, dtype=float),
    }


def draw_past_track(ax, past_track: dict, forecast_start_pt: dict, data_crs):
    if not past_track:
        return

    times = list(past_track["times"])
    lats  = list(past_track["lats"])
    lons  = list(past_track["lons"])
    winds = list(past_track["winds"])

    f_time = forecast_start_pt["time"]
    f_lat  = forecast_start_pt["lat"]
    f_lon  = forecast_start_pt["lon"]
    f_wind = forecast_start_pt["wind"]

    if lons:
        # 修正：past_track 與即將接上的預報起點 f_lon 可能落在不同的換日線分支，
        # 在拼接前先把 past_track 整條平移到與 f_lon 同一分支，
        # 避免畫出一條橫跨整張地圖的離譜連線。
        lons = list(align_lon_to_ref(np.array(lons, dtype=float), f_lon, anchor="last"))

    if not times or times[-1] != f_time:
        times.append(f_time)
        lats.append(f_lat)
        lons.append(f_lon)
        winds.append(f_wind)
    else:
        lats[-1], lons[-1], winds[-1] = f_lat, f_lon, f_wind

    lats  = np.array(lats,  dtype=float)
    lons  = np.array(lons,  dtype=float)
    winds = np.array(winds, dtype=float)

    if len(lats) > 1:
        mid_winds = (winds[:-1] + winds[1:]) / 2
        colors = [get_color(w) for w in mid_winds]
        pts  = np.column_stack([lons, lats])
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        lc   = LineCollection(
            segs, colors=colors, linewidths=2.5, alpha=0.5,
            capstyle="round", transform=data_crs, zorder=3,
        )
        ax.add_collection(lc)

    for i, t in enumerate(times[:-1]):
        hrs_diff = abs((f_time - t).total_seconds() / 3600.0)
        if hrs_diff > 0 and np.isclose(hrs_diff % 24, 0.0, atol=1e-6):
            ax.plot(
                lons[i], lats[i], "o", color=get_color(winds[i]),
                markersize=6, markeredgecolor="black", markeredgewidth=1.0,
                transform=data_crs, zorder=4,
            )


# ══════════════════════════════════════════════════════════════════════════════
# 地圖範圍 (修正：移除 past_track 依據，全以預測路徑為主)
# ══════════════════════════════════════════════════════════════════════════════

def compute_map_params(tracks: list, pad_deg: float = 5.0, target_aspect: float = 4.0 / 3.0):
    if not tracks:
        raise ValueError("compute_map_params: tracks 不可為空")

    ref_lon = float(tracks[0]["lons"][0])

    aligned_lats, aligned_lons = [], []
    for t in tracks:
        lons = np.asarray(t["lons"], dtype=float)
        lats = np.asarray(t["lats"], dtype=float)
        if len(lons) == 0:
            continue
        shift = np.round((ref_lon - lons[0]) / 360.0) * 360.0
        aligned_lons.append(lons + shift)
        aligned_lats.append(lats)

    all_lats = np.concatenate(aligned_lats)
    all_lons = np.concatenate(aligned_lons)

    lat_min, lat_max = float(all_lats.min()), float(all_lats.max())
    lon_min, lon_max = float(all_lons.min()), float(all_lons.max())

    lat_center = (lat_min + lat_max) / 2.0
    lon_center = (lon_min + lon_max) / 2.0

    half_lon_required = (lon_max - lon_min) / 2.0 + pad_deg
    half_lat_required = (lat_max - lat_min) / 2.0 + pad_deg

    half_lon_if_lat_binds = half_lat_required * target_aspect
    half_lat_if_lon_binds = half_lon_required / target_aspect

    if half_lon_required >= half_lon_if_lat_binds:
        half_lon = half_lon_required
        half_lat = half_lat_if_lon_binds
    else:
        half_lat = half_lat_required
        half_lon = half_lon_if_lat_binds

    if half_lat < 4.0:
        half_lat = 4.0
        half_lon = half_lat * target_aspect

    central_lon = lon_center
    extent_proj = (
        lon_center - half_lon, lon_center + half_lon,
        lat_center - half_lat, lat_center + half_lat,
    )
    return central_lon, extent_proj, lon_center, lat_center


# ══════════════════════════════════════════════════════════════════════════════
# 幾何與繪圖
# ══════════════════════════════════════════════════════════════════════════════

def get_geodesic_circle(lon, lat, radius_km, num_points=128):
    if radius_km is None or np.isnan(radius_km) or radius_km <= 0:
        return None
    angles = np.linspace(0, 360, num_points, endpoint=False)
    lons_c, lats_c, _ = _GEOD.fwd(
        [lon] * num_points, [lat] * num_points,
        angles, [radius_km * 1000] * num_points,
    )
    lons_c = np.array(lons_c)
    lons_c = ((lons_c - lon + 180) % 360) + lon - 180
    return Polygon(zip(lons_c, lats_c))


def get_geodesic_sector(lon, lat, radius_km, start_bearing, end_bearing, num_points=64):
    if np.isnan(radius_km) or radius_km <= 0:
        return None
    angles = np.linspace(start_bearing, end_bearing, num_points, endpoint=False)
    lons_c, lats_c, _ = _GEOD.fwd(
        [lon] * num_points, [lat] * num_points,
        angles, [radius_km * 1000] * num_points,
    )
    lons_c = np.array(lons_c)
    lons_c = ((lons_c - lon + 180) % 360) + lon - 180
    sector_ring = [(lon, lat)] + list(zip(lons_c, lats_c))
    return Polygon(sector_ring)


def build_cone_polygon(path_lons, path_lats, path_times, radius_fn,
                       interp_steps: int = 50, smooth_deg: float = 30):
    if len(path_times) < 2:
        return None

    t0 = path_times[0]
    waypoints = [
        (lon, lat, radius_fn(i, (t - t0).total_seconds() / 3600))
        for i, (lon, lat, t) in enumerate(zip(path_lons, path_lats, path_times))
    ]
    if len(waypoints) < 2:
        return None

    circles = []
    for j in range(len(waypoints) - 1):
        lon0, lat0, r0 = waypoints[j]
        lon1, lat1, r1 = waypoints[j + 1]
        for k in range(interp_steps):
            alpha = k / interp_steps
            poly = get_geodesic_circle(
                lon0 + alpha * (lon1 - lon0),
                lat0 + alpha * (lat1 - lat0),
                r0   + alpha * (r1   - r0),
            )
            if poly is not None:
                circles.append(poly)

    lon_f, lat_f, r_f = waypoints[-1]
    last_poly = get_geodesic_circle(lon_f, lat_f, r_f)
    if last_poly is not None:
        circles.append(last_poly)

    if not circles:
        return None

    full_cone = unary_union(circles)
    if not full_cone.is_empty:
        full_cone = full_cone.buffer(smooth_deg, quad_segs=8).buffer(-smooth_deg, quad_segs=8)
    return full_cone if not full_cone.is_empty else None


def _add_cone_to_axes(ax, cone_polygon):
    ax.add_geometries(
        [cone_polygon], crs=ccrs.PlateCarree(),
        facecolor="#F3FF4A", edgecolor="#cc3300", alpha=0.4, zorder=2,
    )


def draw_forecast_cone(ax, track: dict):
    cone = build_cone_polygon(
        track["lons"], track["lats"], track["times"],
        radius_fn=lambda i, hours: get_error_radius(hours),
    )
    if cone is not None:
        _add_cone_to_axes(ax, cone)


def compute_member_spread_radius(tracks: list, time_idx: int, fixed_fallback_km: float) -> float:
    actual_members = [t for t in tracks if t["sample_id"] != -1.0]
    n_total = len(actual_members)
    lats = [t["lats"][time_idx] for t in actual_members if time_idx < len(t["lats"])]
    lons = [t["lons"][time_idx] for t in actual_members if time_idx < len(t["lons"])]

    if len(lats) < n_total / 2 or len(lats) < 2:
        return fixed_fallback_km

    mean_lat, mean_lon = float(np.mean(lats)), float(np.mean(lons))
    max_dist_km = 0.0
    for la, lo in zip(lats, lons):
        _, _, dist_m = _GEOD.inv(mean_lon, mean_lat, lo, la)
        max_dist_km = max(max_dist_km, dist_m / 1000.0)
    return max_dist_km


def draw_member_spread_cone(ax, tracks: list, mean: dict):
    if not mean:
        return

    n_mean_points    = len(mean["lons"])
    last_valid_radius = ERROR_RADII[0]

    def radius_fn(i, hours):
        nonlocal last_valid_radius
        if i >= n_mean_points:
            return last_valid_radius
        fixed   = get_error_radius(hours)
        r       = compute_member_spread_radius(tracks, i, fixed)
        actual_count = len([t for t in tracks if t["sample_id"] != -1.0])
        n_valid = sum(1 for tr in tracks if tr["sample_id"] != -1.0 and i < len(tr["lats"]))
        if n_valid < actual_count / 2:
            return last_valid_radius
        last_valid_radius = r
        return r

    cone = build_cone_polygon(mean["lons"], mean["lats"], mean["times"], radius_fn)
    if cone is not None:
        _add_cone_to_axes(ax, cone)


def draw_track(ax, lons, lats, winds, lw=0.9, alpha=0.70, zorder=2, data_crs=None):
    if data_crs is None:
        data_crs = ccrs.PlateCarree()
    if len(lons) < 2:
        return
    mid_winds = (winds[:-1] + winds[1:]) / 2
    colors = [get_color(w) for w in mid_winds]
    pts  = np.column_stack([lons, lats])
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    lc   = LineCollection(
        segs, colors=colors, linewidths=lw, alpha=alpha,
        capstyle="round", transform=data_crs, zorder=zorder,
    )
    ax.add_collection(lc)


def draw_radius_circles(ax, track: dict):
    valid_polygons = []
    if not track.get("times"):
        return

    t0       = track["times"][0]
    last_idx = len(track["times"]) - 1

    for idx, (time, lon, lat, radius_values) in enumerate(
        zip(track["times"], track["lons"], track["lats"], track["radii"])
    ):
        hours = (time - t0).total_seconds() / 3600.0
        if not np.isclose(hours % 24.0, 0.0, atol=1e-6) and idx != last_idx:
            continue

        for sector_idx, radius_km in enumerate(radius_values):
            sector = get_geodesic_sector(
                float(lon), float(lat), float(radius_km),
                *SECTOR_BEARINGS[sector_idx],
            )
            if sector is not None:
                valid_polygons.append(sector)

    if not valid_polygons:
        return

    ax.add_geometries(
        valid_polygons, crs=ccrs.PlateCarree(),
        facecolor="#003366", edgecolor="#003366",
        linewidth=1.0, alpha=0.15, zorder=4,
    )


def draw_mean_track(ax, mean: dict, data_crs, interval_h: int = 24, show_labels: bool = True):
    lons, lats, winds = mean["lons"], mean["lats"], mean["winds"]
    radii = mean.get("radii")

    ax.plot(lons[0], lats[0], "kx", markersize=11, markeredgewidth=2.5,
            transform=data_crs, zorder=7)

    t0, done = mean["times"][0], set()
    for i, t in enumerate(mean["times"]):
        elh = (t - t0).total_seconds() / 3600.0
        if elh <= 0 or not np.isclose(elh % interval_h, 0.0, atol=1e-3):
            continue

        lh = int(round(elh))
        if lh in done:
            continue
        done.add(lh)

        if show_labels:
            wind_str   = f"{int(round(winds[i]))} kt" if np.isfinite(winds[i]) else ""
            avg_radius = np.nan
            if radii is not None and i < len(radii):
                avg_radius = float(np.nanmean(radii[i]))

            label_lines = [f"{t.strftime('%m-%d %H:%M')} (+{int(lh)} h)"]
            if wind_str and np.isfinite(avg_radius):
                label_lines.append(f"{wind_str} // R={avg_radius:.0f}")
            elif wind_str:
                label_lines.append(wind_str)
            elif np.isfinite(avg_radius):
                label_lines.append(f"R={avg_radius:.0f}")
            label_text = "\n".join(label_lines)

            if i < len(lons) - 1 and i > 0:
                dx, dy = lons[i + 1] - lons[i - 1], lats[i + 1] - lats[i - 1]
            elif i < len(lons) - 1:
                dx, dy = lons[i + 1] - lons[i],     lats[i + 1] - lats[i]
            elif i > 0:
                dx, dy = lons[i] - lons[i - 1],     lats[i] - lats[i - 1]
            else:
                dx, dy = 1.0, 0.0

            norm = np.hypot(dx, dy)
            if not np.isfinite(norm) or norm == 0:
                dx, dy, norm = 1.0, 0.0, 1.0
            nx, ny = -dy / norm, dx / norm

            base_offset  = 2.8
            extra_offset = min(avg_radius / 150.0, 1.2) if np.isfinite(avg_radius) else 0.0
            offset       = base_offset + extra_offset

            def candidate_score(sign: int) -> float:
                cand_lon = lons[i] + nx * sign * offset
                cand_lat = lats[i] + ny * sign * offset
                score    = 0.0
                for ni in range(max(0, i - 2), min(len(lons), i + 3)):
                    if ni == i:
                        continue
                    _, _, dist_m = _GEOD.inv(cand_lon, cand_lat, lons[ni], lats[ni])
                    score += dist_m / 1000.0
                return score

            sign      = 1 if candidate_score(1) >= candidate_score(-1) else -1
            label_lon = lons[i] + nx * sign * offset
            label_lat = lats[i] + ny * sign * offset

            ax.plot(
                lons[i], lats[i], "ko",
                markersize=5, markerfacecolor="white", markeredgecolor="black",
                markeredgewidth=1.2, transform=data_crs, zorder=8,
            )
            ax.annotate(
                label_text,
                xy=(lons[i], lats[i]), xycoords=data_crs,
                xytext=(label_lon, label_lat), textcoords=data_crs,
                arrowprops=dict(arrowstyle="-", color="black", lw=1.1, alpha=0.8),
                fontsize=8.5, fontweight="bold", color="black",
                ha="left" if sign * nx >= 0 else "right", va="center",
                path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
                zorder=8,
            )


def draw_member_list(fig, tracks: list, mean: dict):
    actual_members = [t for t in tracks if t["sample_id"] != -1.0]
    items    = sorted(actual_members, key=lambda t: (t["max_wind"], -t["min_pressure"]), reverse=True)
    ax2      = fig.add_axes([0.843, 0.12, 0.15, 0.8])
    ax2.axis("off")
    draw_mean = mean is not None

    for i in range(len(items) + (1 if draw_mean else 0)):
        y = 1.0 - i / 47
        if i < len(items):
            t     = items[i]
            color = get_color(t["max_wind"])
            ax2.text(0.05, y, f'#{int(t["sample_id"] + 1)}',
                     fontsize=10, color=color, va="center", ha="left", fontfamily="monospace")
            ax2.text(0.25, y, f'{t["max_wind"]:.1f}',
                     fontsize=10, color=color, va="center", ha="left", fontfamily="monospace")
            pressure_str = (
                f'kt  {t["min_pressure"]:.1f}hPa'
                if t["min_pressure"] < 1000
                else f'kt  {int(t["min_pressure"])}hPa'
            )
            ax2.text(0.45, y, pressure_str,
                     fontsize=10, color=color, va="center", ha="left", fontfamily="monospace")
        else:
            color = get_color(mean["max_wind"])
            ax2.text(0.00, y - 0.03, "mean",
                     fontsize=10, color="black", va="center", ha="left", fontfamily="monospace")
            ax2.text(0.25, y - 0.03, f'{mean["max_wind"]:.1f}',
                     fontsize=10, color=color, va="center", ha="left", fontfamily="monospace")
            pressure_str = (
                f'kt  {mean["min_pressure"]:.1f}hPa'
                if mean["min_pressure"] < 1000
                else f'kt  {int(mean["min_pressure"])}hPa'
            )
            ax2.text(0.45, y - 0.03, pressure_str,
                     fontsize=10, color=color, va="center", ha="left", fontfamily="monospace")


# ══════════════════════════════════════════════════════════════════════════════
# 地圖底圖
# ══════════════════════════════════════════════════════════════════════════════

def _build_map_figure(tracks: list, wide: bool = False):
    if wide:
        aspect  = 4.7 / 3.0
        ax_rect = [0.01, 0.04, 0.99, 0.89]
        top_line = [0.039, 0.97], [0.937, 0.937]
    else:
        aspect  = 4.0 / 3.0
        ax_rect = [0.01, 0.04, 0.85, 0.89]
        top_line = [0.039, 0.99], [0.937, 0.937]

    central_lon, extent_proj, _, _ = compute_map_params(tracks, target_aspect=aspect)
    proj     = ccrs.PlateCarree(central_longitude=central_lon)
    data_crs = ccrs.PlateCarree()
    lon_min_e, lon_max_e, lat_min_e, lat_max_e = extent_proj

    def to_standard(lon_uw):
        return ((lon_uw - central_lon + 180) % 360) - 180 + central_lon

    fig = plt.figure(figsize=(13.5, 9), facecolor="white")
    ax  = fig.add_axes(ax_rect, projection=proj)
    ax.set_extent(
        [to_standard(lon_min_e), to_standard(lon_max_e), lat_min_e, lat_max_e],
        crs=data_crs,
    )

    fig.add_artist(matplotlib.lines.Line2D(
        top_line[0], top_line[1],
        transform=fig.transFigure, color="black", lw=1, zorder=10,
    ))

    for feat, kw in _MAP_FEATURES:
        ax.add_feature(feat, **kw)

    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="gray", alpha=0.5,
                      linestyle="--", crs=data_crs)
    gl.top_labels   = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 8}
    gl.ylabel_style = {"size": 8}

    handles = [mpatches.Patch(facecolor=c["color"], label=c["name"]) for c in CATEGORIES]
    ax.legend(handles=handles, loc="lower left", fontsize=8,
              framealpha=0.88, edgecolor="#AAAAAA", facecolor="white")

    return fig, ax, data_crs


# ── 台灣範圍特寫子圖 ─────────────────────────────────────────────────────────

def _add_taiwan_inset(fig, draw_callback):
    data_crs_inset = ccrs.PlateCarree()
    ax_inset = fig.add_axes(TAIWAN_INSET_RECT, projection=data_crs_inset, zorder=20)
    lon_min, lon_max, lat_min, lat_max = TAIWAN_INSET_EXTENT
    ax_inset.set_extent([lon_min, lon_max, lat_min, lat_max], crs=data_crs_inset)

    for feat, kw in _MAP_FEATURES:
        ax_inset.add_feature(feat, **kw)

    _ = ax_inset.gridlines(draw_labels=False, linewidth=0.3, color="gray",
                            alpha=0.5, linestyle="--")

    for spine in ax_inset.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.3)

    draw_callback(ax_inset, data_crs_inset)
    return ax_inset


def _save_plot(fig, output_dir: str, track_id: str, model_name: str,
               init_str: str, is_cone_model: bool) -> str:
    safe_id      = track_id.replace("/", "_").replace("\\", "_")
    out_filename = f"{init_str}_cone.png" if is_cone_model else f"{init_str}.png"
    out_path     = os.path.join(output_dir, safe_id, model_name, out_filename)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=100, bbox_inches=None, facecolor="white")
    plt.close(fig)
    return out_path


def _get_dat_path(track_id: str) -> str | None:
    dat_dir = os.path.join("active_typhoon", track_id)
    if not os.path.isdir(dat_dir):
        return None
    dat_files = [
        f for f in glob.glob(os.path.join(dat_dir, "*.dat"))
        if os.path.basename(f).lower().startswith("b")
    ]
    return sorted(dat_files)[0] if dat_files else None


def _render_main_figure(track_id: str, tracks: list, mean: dict | None,
                        past_track: dict | None, forecast_start_pt: dict,
                        output_dir: str, model_name: str):
    is_cone_model = len(tracks) == 1 or (len(tracks) == 2 and any(t["sample_id"] == -1.0 for t in tracks))
    
    fig, ax, data_crs = _build_map_figure(tracks, wide=is_cone_model)

    if past_track:
        draw_past_track(ax, past_track, forecast_start_pt, data_crs)

    if is_cone_model:
        cone_track = next((t for t in tracks if t["sample_id"] == -1.0), tracks[0])
        draw_forecast_cone(ax, cone_track)
        draw_track(ax, cone_track["lons"], cone_track["lats"], cone_track["winds"],
                   lw=2.5, alpha=1.0, zorder=5, data_crs=data_crs)
        draw_radius_circles(ax, cone_track)
        draw_mean_track(ax, cone_track, data_crs, show_labels=True)
    else:
        for tr in tracks:
            if tr["sample_id"] == -1.0:
                continue
            draw_track(ax, tr["lons"], tr["lats"], tr["winds"],
                       lw=0.9, alpha=0.65, zorder=2, data_crs=data_crs)
        draw_member_list(fig, tracks, mean)
        if mean:
            draw_mean_track(ax, mean, data_crs, show_labels=False)

    init_dt  = forecast_start_pt["time"]
    init_str = init_dt.strftime("%Y-%m-%dT%H")
    current_wind = forecast_start_pt["wind"]
    current_pressure = forecast_start_pt.get("pressure")
    title_text = _build_plot_title(model_name, track_id, current_wind, current_pressure, is_cone_model)

    ax.annotate(
        title_text,
        xy=(0, 1.01), xycoords="axes fraction", textcoords="offset points",
        ha="left", va="bottom", fontsize=13, color="#444444", fontweight="bold",
    )
    ax.annotate(
        f"Made By EGTY\n{init_str}:Forecast time",
        xy=(1.0 if is_cone_model else 1.2, 1.01),
        xycoords="axes fraction", textcoords="offset points",
        ha="right", va="bottom", fontsize=13, color="#444444", fontweight="bold",
    )

    _save_plot(fig, output_dir, track_id, model_name, init_str, is_cone_model)


def _render_spread_figure(track_id: str, tracks: list, mean: dict,
                          past_track: dict | None, forecast_start_pt: dict,
                          output_dir: str, model_name: str):
    actual_members = [t for t in tracks if t["sample_id"] != -1.0]
    if len(actual_members) <= 1 or not mean:
        return

    fig, ax, data_crs = _build_map_figure(tracks, wide=True)

    if past_track:
        draw_past_track(ax, past_track, forecast_start_pt, data_crs)

    draw_member_spread_cone(ax, tracks, mean)
    draw_radius_circles(ax, mean)
    draw_track(ax, mean["lons"], mean["lats"], mean["winds"],
               lw=2.5, alpha=1.0, zorder=5, data_crs=data_crs)
    draw_mean_track(ax, mean, data_crs, show_labels=True)

    init_dt  = forecast_start_pt["time"]
    init_str = init_dt.strftime("%Y-%m-%dT%H")
    current_wind = forecast_start_pt["wind"]
    current_pressure = forecast_start_pt.get("pressure")
    title_text = f"{model_name} [{track_id}] Member Spread Plot\n" + _format_current_intensity(current_wind, current_pressure)

    ax.annotate(
        title_text,
        xy=(0, 1.01), xycoords="axes fraction", textcoords="offset points",
        ha="left", va="bottom", fontsize=13, color="#444444", fontweight="bold",
    )
    ax.annotate(
        f"Made By EGTY\n{init_str}:Forecast time",
        xy=(1.0, 1.01), xycoords="axes fraction", textcoords="offset points",
        ha="right", va="bottom", fontsize=13, color="#444444", fontweight="bold",
    )

    _save_plot(fig, output_dir, track_id, model_name, init_str, is_cone_model=True)


# ══════════════════════════════════════════════════════════════════════════════
# 清理舊圖 / 合併
# ══════════════════════════════════════════════════════════════════════════════

def cleanup_old_tracks(output_dir: str):
    """
    清理舊檔案邏輯：
    1. 每個模型的圖片如果超過 40 張，依然會修剪（保留最新的 40 張）。
    2. 對於「整個風暴目錄」的過期清理，只有當【該風暴內的所有模型目錄】都超過 24 小時未更新或為空時，才會一起整區塊刪除。
    """
    if not os.path.exists(output_dir):
        return

    now = datetime.now()

    for track_id in os.listdir(output_dir):
        track_path = os.path.join(output_dir, track_id)
        if not os.path.isdir(track_path):
            continue

        model_dirs = [os.path.join(track_path, m) for m in os.listdir(track_path) if os.path.isdir(os.path.join(track_path, m))]
        
        # 建立狀態記錄：用來追蹤此風暴下是否還有「任何一個模型」處於活躍狀態 (24h內更新過)
        has_any_active_model = False
        has_models = len(model_dirs) > 0

        for model_dir in model_dirs:
            model_name = os.path.basename(model_dir)
            png_files = sorted(glob.glob(os.path.join(model_dir, "*.png")))
            
            if not png_files:
                continue

            # ── 內部保留機制：修剪個別模型組別超過 40 張的舊圖 ──
            base_files    = sorted(f for f in png_files if not f.endswith("_cone.png") and not f.endswith("_compare.png"))
            cone_files    = sorted(f for f in png_files if f.endswith("_cone.png"))
            compare_files = sorted(f for f in png_files if f.endswith("_compare.png"))

            for group in (base_files, cone_files, compare_files):
                if len(group) > 40:
                    for f in group[:-40]:
                        try:
                            os.remove(f)
                            print(f"🗑️ [{model_name}] 刪除舊圖 (超過40張): {f}")
                        except Exception as e:
                            print(f"⚠️ 刪除失敗 {f}: {e}")

            # 重新獲取修剪後的圖片清單
            remaining_imgs = sorted(glob.glob(os.path.join(model_dir, "*.png")))
            if not remaining_imgs:
                continue

            # 從最新一張圖檔名中提取時間戳記
            newest_img = remaining_imgs[-1]
            newest_name = os.path.basename(newest_img).replace("_cone.png", "").replace("_compare.png", "").replace(".png", "")
            
            try:
                newest_time = datetime.strptime(newest_name, "%Y-%m-%dT%H")
                # 如果小於或等於 24 小時，認定該模型依然活耀
                if (now - newest_time).total_seconds() <= 24 * 3600:
                    has_any_active_model = True
            except ValueError:
                # 檔名若解析失敗（例如非標準預報時間格式），保險起見將其視為活耀
                print(f"⚠️ 無法從檔名解析時間: {newest_name}，默認保留")
                has_any_active_model = True

        # ── 風暴整體清理判定 ──
        # 只有在「完全沒有任何活躍模型」且「有模型目錄」的情況下，才視為整個風暴完全過期並一起清理
        if has_models and not has_any_active_model:
            try:
                shutil.rmtree(track_path)
                print(f"🧹 氣旋風暴 [{track_id}] 所有模型均已超過 24h 未更新，已整區塊自動清理。")
            except Exception as e:
                print(f"⚠️ 刪除風暴目錄 {track_path} 失敗: {e}")


def standardise_typhoon_id(raw_id: str) -> str:
    s = raw_id.strip().upper()
    if s.startswith("BWP"):
        s = s[1:]
    match = re.match(r"^([A-Z]{2})([0-9A-Z]{2})([0-9]{4})$", s)
    if match:
        basin, num_code, year = match.groups()
        if num_code.startswith("B") and num_code[1].isdigit():
            num_code = f"9{num_code[1]}"
        elif num_code.startswith("A") and num_code[1].isdigit():
            num_code = f"8{num_code[1]}"
        return f" {basin}{num_code}{year}"
    return s


def find_transition_in_content(content_bytes: bytes):
    try:
        text  = content_bytes.decode("utf-8")
        match = re.search(
            r"TRANSITIONED\s*,\s*([a-zA-Z0-9]+)\s+to\s+([a-zA-Z0-9]+)", text, re.I
        )
        if match:
            src, dest = match.groups()
            return standardise_typhoon_id(src), standardise_typhoon_id(dest)
    except Exception:
        pass
    return None


def _merge_track_folders(src_folder: str, dest_folder: str, src_id: str, dest_id: str):
    if not os.path.isdir(src_folder):
        return
    for model_name in os.listdir(src_folder):
        src_model_dir  = os.path.join(src_folder, model_name)
        dest_model_dir = os.path.join(dest_folder, model_name)
        if not os.path.isdir(src_model_dir):
            continue
        os.makedirs(dest_model_dir, exist_ok=True)
        for src_file in glob.glob(os.path.join(src_model_dir, "*.png")):
            fname     = os.path.basename(src_file)
            dest_file = os.path.join(dest_model_dir, fname)
            if os.path.exists(dest_file):
                if os.path.getmtime(src_file) > os.path.getmtime(dest_file):
                    shutil.copy2(src_file, dest_file)
                    print(f"  ♻️  [{model_name}] {fname}：來源較新，已覆蓋")
                else:
                    print(f"  ✔️  [{model_name}] {fname}：目標已是最新，略過")
            else:
                shutil.copy2(src_file, dest_file)
                print(f"  📂  [{model_name}] {fname}：已複製至 {dest_id}")


def merge_same_storm_output_folders(output_dir: str, active_dir: str = "active_typhoon"):
    if not os.path.exists(active_dir) or not os.path.exists(output_dir):
        return

    transitions: dict[str, str] = {}
    for folder in os.listdir(active_dir):
        folder_path = os.path.join(active_dir, folder)
        if not os.path.isdir(folder_path) or folder == "model_tracks":
            continue
        for file in os.listdir(folder_path):
            if file.endswith(".dat"):
                with open(os.path.join(folder_path, file), "rb") as f:
                    content = f.read()
                trans = find_transition_in_content(content)
                if trans:
                    src, dest = trans
                    transitions[dest] = src
                    print(f"\n🔗 偵測到氣旋升級過渡（PNG合併用）：{src} -> {dest}")

    if not transitions:
        print("  （無需合併的過渡關係）")
        return

    for dest_id, src_id in transitions.items():
        dest_folder = os.path.join(output_dir, dest_id)
        src_folder  = os.path.join(output_dir, src_id)
        if not os.path.exists(dest_folder):
            print(f"  ⚠️  目標資料夾不存在，略過：{dest_folder}")
            continue
        if not os.path.exists(src_folder):
            print(f"  ⚠️  來源資料夾不存在，略过：{src_folder}")
            continue
        print(f"\n📂 合併 PNG：{src_id} → {dest_id}")
        _merge_track_folders(src_folder, dest_folder, src_id, dest_id)
        try:
            shutil.rmtree(src_folder)
            print(f"  橫 🧹 已刪除來源 PNG 資料夾：{src_folder}")
        except Exception as e:
            print(f"  ⚠️  無法刪除 {src_folder}：{e}")


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════

def plot_one_track(track_id: str, samples: list, output_dir: str, model_name: str = "MODEL"):
    cone_samples   = [s for s in samples if s.get("sample_id") == -1.0]
    member_samples = [s for s in samples if s.get("sample_id") != -1.0]

    if cone_samples:
        tracks = parse_tracks(cone_samples)
        if tracks:
            init_dt = tracks[0]["times"][0]
            past_track = parse_best_track_dat(_get_dat_path(track_id), init_dt) \
                         if _get_dat_path(track_id) else None
            forecast_start_pt = {
                "time":     init_dt,
                "lat":      tracks[0]["lats"][0],
                "lon":      tracks[0]["lons"][0],
                "wind":     tracks[0]["winds"][0],
                "pressure": tracks[0]["pressure"][0],
            }
            _render_main_figure(track_id, tracks, None, past_track,
                                forecast_start_pt, output_dir, model_name)

    if member_samples:
        tracks = parse_tracks(samples)
        if not tracks:
            print(f"  [SKIP] {track_id}：無有效軌跡資料")
            return

        ref_track = next((t for t in tracks if t["sample_id"] != -1.0))
        init_dt    = ref_track["times"][0]
        past_track = parse_best_track_dat(_get_dat_path(track_id), init_dt) \
                     if _get_dat_path(track_id) else None

        mean = ensemble_mean(tracks)

        forecast_start_pt = {
            "time":     init_dt,
            "lat":      mean["lats"][0]     if mean else ref_track["lats"][0],
            "lon":      mean["lons"][0]     if mean else ref_track["lons"][0],
            "wind":     mean["winds"][0]    if mean else ref_track["winds"][0],
            "pressure": mean["pressure"][0] if mean else ref_track["pressure"][0],
        }

        _render_main_figure(track_id, tracks, mean, past_track,
                            forecast_start_pt, output_dir, model_name)
        _render_spread_figure(track_id, tracks, mean, past_track,
                              forecast_start_pt, output_dir, model_name)


def run_model_pipeline(model_name: str, output_dir: str):
    cfg = MODEL_CONFIGS[model_name]
    print(f"\n{'='*60}\n  模型：{model_name}\n{'='*60}")

    if not os.path.exists(cfg["json_path"]):
        print(f"❌ [{model_name}] 找不到 {cfg['json_path']}，跳過繪圖。")
        return

    data = load_json(cfg["json_path"])
    json_updated = False

    for track_id, samples in data.items():
        has_average = any(s.get("sample_id") == -1.0 for s in samples)
        if not has_average and len([s for s in samples if s.get("sample_id") != -1.0]) > 1:
            avg_sample = calculate_average_track(samples)
            if avg_sample:
                samples.insert(0, avg_sample)
                json_updated = True
                print(f"   ✅ [{model_name}] {track_id} 已計算平均路徑並準備持久化寫回")

    if json_updated:
        try:
            save_json(data, cfg["json_path"])
            print(f"   💾 [{model_name}] 成功將新計算的平均路徑寫回 {cfg['json_path']}")
        except Exception as e:
            print(f"   ⚠️ [{model_name}] 寫回 JSON 失敗: {e}")

    print(f"🌀 [{model_name}] 共 {len(data)} 個 track_id，開始繪圖…")
    for track_id, samples in data.items():
        plot_one_track(track_id, samples, output_dir, model_name=model_name)
    print(f"🎉 [{model_name}] 繪圖完成")


# ── 多模型平均路徑比較圖 ──────────────────────────────────────────────────────

def _draw_multimodel_track(ax, track: dict, model_name: str, data_crs,
                           interval_h: int = 6, show_labels: bool = True):
    color = MODEL_COLORS.get(model_name, "#333333")
    lons, lats = track["lons"], track["lats"]

    if len(lons) < 2:
        return

    pts  = np.column_stack([lons, lats])
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    lc   = LineCollection(
        segs, colors=[color] * len(segs), linewidths=2.5, alpha=0.9,
        capstyle="round", transform=data_crs, zorder=5,
    )
    ax.add_collection(lc)

    ax.plot(lons[0], lats[0], "x", color=color,
            markersize=11, markeredgewidth=2.5, transform=data_crs, zorder=7)

    t0, done = track["times"][0], set()
    for i, t in enumerate(track["times"]):
        elh = (t - t0).total_seconds() / 3600.0
        if elh <= 0 or not np.isclose(elh % interval_h, 0.0, atol=1e-3):
            continue
        lh = int(round(elh))
        if lh in done:
            continue
        done.add(lh)

        is_day_mark = (lh % 24 == 0)

        if is_day_mark:
            ax.plot(
                lons[i], lats[i], "s",
                markersize=8, markerfacecolor=color,
                markeredgecolor="black", markeredgewidth=1.2,
                transform=data_crs, zorder=8,
            )

            if not show_labels:
                continue

            label_text = f"+{lh}h"

            if i < len(lons) - 1 and i > 0:
                dx, dy = lons[i + 1] - lons[i - 1], lats[i + 1] - lats[i - 1]
            elif i < len(lons) - 1:
                dx, dy = lons[i + 1] - lons[i],     lats[i + 1] - lats[i]
            elif i > 0:
                dx, dy = lons[i] - lons[i - 1],     lats[i] - lats[i - 1]
            else:
                dx, dy = 1.0, 0.0

            norm = np.hypot(dx, dy)
            if not np.isfinite(norm) or norm == 0:
                dx, dy, norm = 1.0, 0.0, 1.0
            nx, ny = -dy / norm, dx / norm

            offset = 2.2

            def _score(sign: int) -> float:
                cand_lon = lons[i] + nx * sign * offset
                cand_lat = lats[i] + ny * sign * offset
                score    = 0.0
                for ni in range(max(0, i - 2), min(len(lons), i + 3)):
                    if ni == i:
                        continue
                    _, _, dist_m = _GEOD.inv(cand_lon, cand_lat, lons[ni], lats[ni])
                    score += dist_m / 1000.0
                return score

            sign      = 1 if _score(1) >= _score(-1) else -1
            label_lon = lons[i] + nx * sign * offset
            label_lat = lats[i] + ny * sign * offset

            ax.annotate(
                label_text,
                xy=(lons[i], lats[i]),     xycoords=data_crs,
                xytext=(label_lon, label_lat), textcoords=data_crs,
                arrowprops=dict(arrowstyle="-", color=color, lw=1.0, alpha=0.8),
                fontsize=7.5, fontweight="bold", color=color,
                ha="left" if sign * nx >= 0 else "right", va="center",
                path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
                zorder=9,
            )
        else:
            ax.plot(
                lons[i], lats[i], "^",
                markersize=5.5, markerfacecolor=color,
                markeredgecolor="black", markeredgewidth=0.8,
                transform=data_crs, zorder=8,
            )


def _render_multimodel_figure(track_id: str,
                              model_tracks: dict[str, dict],
                              past_track: dict | None,
                              forecast_start_pt: dict,
                              output_dir: str,
                              init_str: str):
    if not model_tracks:
        return

    # 修正：多個模型各自的平均路徑可能落在不同換日線分支
    # （例如 FNV3 展開成 179→181，GFSE 展開成 -181→-179），
    # 在合併計算地圖範圍、互相繪製比較前，統一對齊到同一分支。
    all_tracks = align_tracks_lon(list(model_tracks.values()))

    fig, ax, data_crs = _build_map_figure(all_tracks, wide=True)

    if past_track:
        draw_past_track(ax, past_track, forecast_start_pt, data_crs)

    for model_name, track in model_tracks.items():
        _draw_multimodel_track(ax, track, model_name, data_crs)

    if is_in_taiwan_range(all_tracks):
        def _inset_draw_multimodel(ax_i, crs_i):
            if past_track:
                draw_past_track(ax_i, past_track, forecast_start_pt, crs_i)
            for m_name, m_track in model_tracks.items():
                _draw_multimodel_track(ax_i, m_track, m_name, crs_i, show_labels=False)

        _add_taiwan_inset(fig, _inset_draw_multimodel)
        print(f"   ℹ️  [MULTIMODEL] {track_id} 偵測到路徑進入台灣範圍，已開啟特寫子圖。")
    else:
        print(f"   ℹ️  [MULTIMODEL] {track_id} 未經過台灣周邊，不顯示特寫子圖。")

    model_handles = [
        mpatches.Patch(facecolor=MODEL_COLORS.get(m, "#333333"), label=m)
        for m in model_tracks
    ]
    marker_handles = [
        matplotlib.lines.Line2D(
            [], [], marker="s", linestyle="None",
            markersize=8, markerfacecolor="#888888", markeredgecolor="black",
            markeredgewidth=1.0, label="+24h interval",
        ),
        matplotlib.lines.Line2D(
            [], [], marker="^", linestyle="None",
            markersize=8, markerfacecolor="#888888", markeredgecolor="black",
            markeredgewidth=0.8, label="+6h interval",
        ),
    ]
    leg1 = ax.legend(
        handles=model_handles,
        loc="lower left", fontsize=9,
        title="Model", title_fontsize=9,
        framealpha=0.9, edgecolor="#AAAAAA", facecolor="white",
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=marker_handles,
        loc="lower right", fontsize=9,
        title="Marker", title_fontsize=9,
        framealpha=0.88, edgecolor="#AAAAAA", facecolor="white",
    )

    current_wind     = forecast_start_pt["wind"]
    current_pressure = forecast_start_pt.get("pressure")
    subtitle         = _format_current_intensity(current_wind, current_pressure)
    title_text       = (
        f"Multi-Model Mean Track Comparison [{track_id}]\n{subtitle}"
    )
    ax.annotate(
        title_text,
        xy=(0, 1.01), xycoords="axes fraction", textcoords="offset points",
        ha="left", va="bottom", fontsize=13, color="#444444", fontweight="bold",
    )
    ax.annotate(
        f"Made By EGTY\n{init_str}:Forecast time",
        xy=(1.0, 1.01), xycoords="axes fraction", textcoords="offset points",
        ha="right", va="bottom", fontsize=13, color="#444444", fontweight="bold",
    )

    safe_id  = track_id.replace("/", "_").replace("\\", "_")
    out_path = os.path.join(output_dir, safe_id, "MULTIMODEL", f"{init_str}_compare.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches=None, facecolor="white")
    plt.close(fig)
    print(f"  ✅  [MULTIMODEL] {track_id} compare  →  {out_path}")


def collect_mean_tracks_across_models(output_dir: str) -> dict[str, dict[str, dict]]:
    result: dict[str, dict[str, dict]] = {}

    for model_name, cfg in MODEL_CONFIGS.items():
        if not os.path.exists(cfg["json_path"]):
            continue

        data = load_json(cfg["json_path"])
        for track_id, samples in data.items():
            cone_samples = [s for s in samples if s.get("sample_id") == -1.0]
            if not cone_samples:
                member_samples = [s for s in samples if s.get("sample_id") != -1.0]
                if len(member_samples) > 1:
                    avg = calculate_average_track(member_samples)
                    if avg:
                        cone_samples = [avg]

            if not cone_samples:
                continue

            tracks = parse_tracks(cone_samples)
            if not tracks:
                continue

            result.setdefault(track_id, {})[model_name] = tracks[0]

    return {tid: models for tid, models in result.items() if len(models) >= 2}


def run_multimodel_pipeline(output_dir: str):
    print(f"\n{'='*60}\n  多模型比較圖\n{'='*60}")

    multimodel_data = collect_mean_tracks_across_models(output_dir)
    if not multimodel_data:
        print("  ℹ️  無符合條件的 track_id（需 ≥2 個模型），跳過多模型比較圖。")
        return

    for track_id, model_tracks in multimodel_data.items():
        # 以各模型 init time 中「最新」的一個為基準時間，
        # 並濾掉比這個最新時間早超過 24 小時（視為未更新）的模型。
        latest_init_dt = max(t["times"][0] for t in model_tracks.values())
        fresh_model_tracks = {
            m: t for m, t in model_tracks.items()
            if (latest_init_dt - t["times"][0]).total_seconds() <= 24 * 3600
        }

        stale_models = set(model_tracks) - set(fresh_model_tracks)
        if stale_models:
            print(
                f"   ℹ️  [MULTIMODEL] {track_id} 排除超過24h未更新模型："
                f"{', '.join(sorted(stale_models))}"
            )

        if len(fresh_model_tracks) < 2:
            print(f"   ℹ️  [MULTIMODEL] {track_id} 過濾後剩餘模型數 < 2，跳過比較圖。")
            continue

        # 取「最新初始時間」對應的那個模型 track，作為命名與 forecast_start_pt 基準
        ref_track = next(
            t for t in fresh_model_tracks.values() if t["times"][0] == latest_init_dt
        )
        init_dt  = latest_init_dt
        init_str = init_dt.strftime("%Y-%m-%dT%H")

        past_track = parse_best_track_dat(_get_dat_path(track_id), init_dt) \
                     if _get_dat_path(track_id) else None

        forecast_start_pt = {
            "time":     init_dt,
            "lat":      ref_track["lats"][0],
            "lon":      ref_track["lons"][0],
            "wind":     ref_track["winds"][0],
            "pressure": ref_track["pressure"][0],
        }

        _render_multimodel_figure(
            track_id, fresh_model_tracks, past_track,
            forecast_start_pt, output_dir, init_str,
        )

    print(f"🎉 多模型比較圖完成，共 {len(multimodel_data)} 個風暴")


def main():
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "active_typhoon")
    os.makedirs(output_dir, exist_ok=True)

    for model_name in MODEL_CONFIGS:
        run_model_pipeline(model_name, output_dir)

    run_multimodel_pipeline(output_dir)

    print(f"\n{'='*60}")
    print("🔗 開始合併相同風暴 PNG 資料夾…")
    merge_same_storm_output_folders(output_dir, active_dir="active_typhoon")
    print("✅ 合併完成！")

    print(f"\n{'='*60}")
    print("🧹 開始自動清理…")
    cleanup_old_tracks(output_dir)
    print("✅ 清理完成！")


if __name__ == "__main__":
    main()
