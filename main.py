print("hi")
import math
import re
import os
import json
import time
import requests
import numpy as np
import xarray as xr
import pandas as pd
from herbie import Herbie
from urllib.parse import urljoin
from datetime import datetime, timedelta, timezone

# =============================================================================
# 全域設定
# =============================================================================
LAND_MASK_PATH  = "environment-2down/land_mask.npy"
OUTPUT_NPZ      = "V4_model_data.npz"
STATE_FILE      = "forecast_list.json"   # 記錄上次成功的週期
DOWNSAMPLE_FACTOR = 2
GRAVITY           = 9.80665

TARGET_VAR_ORDER = [
    't2m', 'msl', 'u10', 'v10',
    'sst',
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

# cfgrib 篩選鍵對照表
GRIB_MAP = {
    't2m': {'shortName': '2t'},
    'msl': {'shortName': 'prmsl'},
    'u10': {'shortName': '10u'},
    'v10': {'shortName': '10v'},
    'sst': {'shortName': 't', 'typeOfLevel': 'surface'},
    't':   {'shortName': 't'},
    'z':   {'shortName': 'gh'},
    'u':   {'shortName': 'u'},
    'v':   {'shortName': 'v'},
    'q':   {'shortName': 'q'},
}

# GRIB2 下載篩選語法
SEARCH_PATTERN = (
    ":("
    "TMP:2 m above ground|"
    "PRMSL:mean sea level|"
    "UGRD:10 m above ground|"
    "VGRD:10 m above ground|"
    "TMP:surface|"
    "(HGT|UGRD|VGRD|TMP):850 mb|"
    "(HGT|UGRD|VGRD|SPFH):500 mb|"
    "(UGRD|VGRD):200 mb"
    ")"
)

# ── 模型設定：新增模型在此擴充即可 ─────────────────────────────────────────────
MODEL_CONFIGS = {
    "FNV3": {
        "json_path":   "active_typhoon/cyclone_data_fnv3.json",
        "csv_path":    "active_typhoon/FNV3.csv",
        "url_template": (
            "https://deepmind.google.com/science/weatherlab/download/cyclones/"
            "FNV3/ensemble/paired/csv/FNV3_{time}_paired.csv"
        ),
        "time_offset_h": 0,
    },
    "GENC": {
        "json_path":   "active_typhoon/cyclone_data_genc.json",
        "csv_path":    "active_typhoon/GENC.csv",
        "url_template": (
            "https://deepmind.google.com/science/weatherlab/download/cyclones/"
            "GENC/ensemble/paired/csv/GENC_{time}_paired.csv"
        ),
        "time_offset_h": 0,
    },
}


# =============================================================================
# 工具函式
# =============================================================================

def get_latest_available_cycle():
    """計算目前 UTC 時間下最新可用的 GFS 初始場週期。"""
    buffer = timedelta(hours=4)
    now    = datetime.now(timezone.utc) - buffer
    for cycle_h in [18, 12, 6, 0]:
        if now.hour >= cycle_h:
            return now.strftime("%Y%m%d"), f"{cycle_h:02d}"
    prev = now - timedelta(days=1)
    return prev.strftime("%Y%m%d"), "18"


def get_previous_cycle(date_str, cycle_str):
    """取得前一個 GFS 週期（往前 6 小時）。"""
    t = datetime.strptime(f"{date_str}{cycle_str}", "%Y%m%d%H").replace(tzinfo=timezone.utc)
    p = t - timedelta(hours=6)
    return p.strftime("%Y%m%d"), p.strftime("%H")


def cycle_id(date_str, cycle_str):
    return f"{date_str}_{cycle_str}Z"


def get_forecast_list(json_path="forecast_list.json"):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            v4_list = data.get("EGTY_V4", [])
            folders = []
            for item in v4_list:
                folders.append(item["folder"].replace("-", "").replace("T", "_") + "Z")
            return folders
    except json.JSONDecodeError:
        print("讀取json錯誤")
        return []
    
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


def _safe_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_quadrant_radii(row):
    keys = {
        "NE": "radius_34_knot_winds_ne_km",
        "SE": "radius_34_knot_winds_se_km",
        "SW": "radius_34_knot_winds_sw_km",
        "NW": "radius_34_knot_winds_nw_km",
    }
    radii = {}
    any_value = False
    for label, key in keys.items():
        value = row.get(key)
        value = _safe_float(value)
        radii[label] = value
        if value is not None:
            any_value = True
    return radii if any_value else None


def _build_data_point(row):
    point = {
        "valid_time": row["valid_time"],
        "coordinates": {
            "lat": _safe_float(row["lat"]),
            "lon": _safe_float(row["lon"]),
        },
        "intensity": {
            "mslp_hpa":       _safe_float(row["minimum_sea_level_pressure_hpa"]),
            "max_wind_knots": _safe_float(row["maximum_sustained_wind_speed_knots"]),
        },
    }
    radii = _extract_quadrant_radii(row)
    if radii is not None:
        point["intensity"]["r34_quadrant_km"] = radii
    return point

# =============================================================================
# 資料下載
# =============================================================================

def download_gfs(date, cycle, hour="000"):
    """使用 Herbie 下載指定 GFS 週期的篩選 GRIB 檔。"""
    tag = f"{date} {cycle}Z F{hour}"
    print(f"[下載] {tag}")
    H = Herbie(
        date=f"{date} {cycle}:00",
        model='gfs',
        product='pgrb2.0p25',
        fxx=int(hour),
        save_dir="./herbie_cache",
    )
    try:
        path = H.download(search=SEARCH_PATTERN)
    except Exception as e:
        print(f"[錯誤] Herbie 下載失敗: {e}")
        return None

    if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
        print(f"[錯誤] 下載的檔案無效: {path}")
        if path and os.path.exists(path):
            os.remove(path)
        return None

    print(f"[完成] {path} ({os.path.getsize(path)/1e6:.1f} MB)")
    return path

def download_model_data(model_name: str) -> str | None:
    cfg = MODEL_CONFIGS[model_name]

    if not os.path.exists("forecast_list.json"):
        print(f"❌ [{model_name}] 找不到 forecast_list.json，跳過下載。")
        return None

    file_time = load_json("forecast_list.json").get("EGTY_V4")[0].get("folder")
    dt_obj = parse_dt(file_time)
    if not dt_obj:
        print(f"❌ [{model_name}] 無法解析時間字串: {file_time}")
        return None

    dt_obj += timedelta(hours=cfg["time_offset_h"])
    save_path = cfg["csv_path"]

    max_time_steps = 4  # 往前搜尋 4 個氣象時段 (24小時)
    for step in range(max_time_steps):
        url_time_str = dt_obj.strftime("%Y_%m_%dT%H_00")
        url = cfg["url_template"].format(time=url_time_str)
        
        print(f"🔗 [{model_name}] 嘗試下載時段 {url_time_str}")

        # ─── 新增：原地重試 5 次的機制 ───
        max_network_retries = 5
        success = False
        
        for attempt in range(1, max_network_retries + 1):
            print(f"   📥 發送請求 (嘗試第 {attempt}/{max_network_retries} 次): {url}")
            try:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
                    "Accept-Language": "zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7"
                }

                resp = requests.get(url, headers=headers, timeout=60)
                resp.raise_for_status()

                content_type = resp.headers.get("Content-Type", "").lower()
                response_text_start = resp.text[:100].strip().lower()
                
                if "text/html" in content_type or response_text_start.startswith("<html") or response_text_start.startswith("<!doc"):
                    raise ValueError("抓到 HTML 錯誤頁面而非真實 CSV")
                    
                if not resp.content or len(resp.content.strip()) == 0:
                    raise ValueError("下載到空檔案")

                with open(save_path, "wb") as f:
                    f.write(resp.content)
                
                success = True
                return save_path  # 下載成功，直接回傳路徑

            except Exception as e:
                print(f"   ⚠️ 第 {attempt} 次嘗試失敗，原因: {e}")
                if attempt < max_network_retries:
                    print("   ⏳ 等待 60 秒後重新嘗試...")
                    time.sleep(60)  # 每次失敗等待 1 分鐘
                else:
                    print(f"   ❌ 第 {attempt} 次嘗試均失敗。")
        
        # 如果 5 次重試都失敗了，或者是該時段真的沒資料，則往前推 6 小時試下一個時段
        print(f"⚠️ [{model_name}] 時間點 {url_time_str} 無法獲取有效資料，嘗試往前一個氣象時段...")
        dt_obj -= timedelta(hours=6)

    # ─── 觸發 GitHub Email 機制 ───
    # 當跑完最近 24 小時的所有時段，且「每一次的 5 次重試」全部都失敗時，拋出 Exception 讓 GitHub Actions 中斷並寄信
    raise RuntimeError(f"❌ [{model_name}] 已嘗試過最近 24 小時的所有氣象時段，且每時段重試 5 次皆失敗。程式強制終止！")

def convert_csv_to_json(csv_path: str, output_path: str, model_name: str = ""):
    tag = f"[{model_name}] " if model_name else ""

    df = pd.read_csv(csv_path, skiprows=6)
    df = df.where(pd.notnull(df), None)

    result = {}
    for track_id, track_group in df.groupby("track_id"):
        samples = []
        for sample_id, sample_group in track_group.groupby("sample"):
            sample_data = {"sample_id": float(sample_id), "data_points": []}
            for _, row in sample_group.iterrows():
                sample_data["data_points"].append(_build_data_point(row))
            samples.append(sample_data)

        result[track_id] = samples

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    print(f"✅ {tag}JSON 已儲存至：{output_path}")
    print(f"{'='*60}")


# =============================================================================
# GRIB 解析
# =============================================================================

def parse_grib(filepath):
    """從 GRIB 檔逐一讀取 TARGET_VAR_ORDER 中的變數並堆疊為陣列。"""
    arrays, names = [], []
    lats = lons = time_val = None

    for var_name in TARGET_VAR_ORDER:
        base = var_name.split('_')[0]
        fk   = GRIB_MAP[base].copy()
        if '_' in var_name:
            fk['typeOfLevel'] = 'isobaricInhPa'
            fk['level']       = int(var_name.split('_')[1])

        ds = None
        try:
            ds  = xr.open_dataset(filepath, engine='cfgrib',
                                  backend_kwargs={'indexpath': '',
                                                  'filter_by_keys': fk})
            key = list(ds.data_vars)[0]
            arr = ds[key].squeeze().values

            if var_name.startswith('z_'):
                arr = arr * GRAVITY

            arrays.append(arr)
            names.append(var_name)

            if lats is None:
                lats     = ds['latitude'].values
                lons     = ds['longitude'].values
                time_val = ds['time'].values

            print(f"  ✓ {var_name}")
        except Exception as e:
            print(f"  ✗ {var_name}: {e}")
        finally:
            if ds is not None:
                ds.close()

    if not arrays:
        return None, None, None, None, None

    return np.stack(arrays, axis=0), lats, lons, time_val, names


# =============================================================================
# ATCF 解析工具
# =============================================================================

def parse_atcf_to_json(file_content, typhoon_id):
    """解析 ATCF .dat 檔，回傳最新一個預報週期中，最接近實際觀測(如 tau=0)的資料。"""
    if not file_content:
        return None
        
    lines = [line.strip() for line in file_content.decode('utf-8').split('\n') if line.strip()]
    
    latest_cycle = None
    best_obs = None
    
    # 由後往前掃描
    for line in reversed(lines):
        columns = [item.strip() for item in line.split(',')]
        if len(columns) < 10:
            continue
            
        current_cycle = columns[2]  # 報時，例如 "2026052118"
        
        # 建立基準：第一筆抓到的就是最新週期
        if latest_cycle is None:
            latest_cycle = current_cycle
            
        # 如果讀到比最新週期更舊的資料，直接中斷
        if current_cycle != latest_cycle:
            break
            
        try:
            tau = int(columns[5])
            wind_raw = columns[8]
            mslp_raw = columns[9]
            
            if not (wind_raw.isdigit() and mslp_raw.isdigit()):
                continue
                
            # 解析經緯度
            raw_lat = columns[6]
            lat_val = "".join(filter(str.isdigit, raw_lat))
            lat = float(lat_val) / 10.0
            if 'S' in raw_lat: lat = -lat
            
            raw_lon = columns[7]
            lon_val = "".join(filter(str.isdigit, raw_lon))
            lon = float(lon_val) / 10.0
            if 'W' in raw_lon: lon = -lon
            
            wind = float(wind_raw)
            mslp = float(mslp_raw)
            
            name = "INVEST"
            if len(columns) > 27 and columns[27].strip():
                name = columns[27].strip()

            formatted_time = f"{current_cycle[:4]}-{current_cycle[4:6]}-{current_cycle[6:8]} {current_cycle[8:10]}:00:00"

            # 儲存資料，我們傾向拿 tau=0 (當前定位)，如果沒有，拿最新週期裡時效最短的
            if best_obs is None or abs(tau) < abs(best_obs['_tau']):
                best_obs = {
                    "name": name,
                    "valid_time": formatted_time,
                    "coordinates": {"lat": lat, "lon": lon},
                    "intensity": {
                        "mslp_hpa": mslp,
                        "max_wind_knots": wind
                    },
                    "_tau": tau # 內部輔助比對用
                }
        except Exception:
            continue
            
    if best_obs:
        best_obs.pop('_tau', None) # 移除輔助欄位
        return best_obs
        
    return None

# =============================================================================
# AP 系集解析
# =============================================================================

def _parse_latlon(raw: str) -> float:
    val = float("".join(filter(str.isdigit, raw))) / 10.0
    return -val if raw[-1] in "SW" else val


def parse_atcf_ap_ensemble(file_content: bytes, typhoon_id: str) -> dict | None:
    """
    解析 .dat 中的 APnn 系集成員路徑。
    如果最新的預報週期沒有 AP 資料，會自動往前嘗試舊的週期（至多往前 24 小時）。
    """
    if not file_content:
        return None

    lines = [line.strip() for line in file_content.decode("utf-8").splitlines() if line.strip()]
    
    # 1. 先找出這份檔案中所有出現過的預報週期 (cols[2])
    all_cycles = []
    for line in reversed(lines):
        cols = [c.strip() for c in line.split(",")]
        if len(cols) < 10:
            continue
        cycle = cols[2]
        if cycle not in all_cycles:
            all_cycles.append(cycle)
            
    if not all_cycles:
        return None

    # 最新出現的週期作為基準時間
    file_latest_cycle = all_cycles[0]
    try:
        base_dt = datetime.strptime(file_latest_cycle, "%Y%m%d%H")
    except Exception:
        return None

    # 2. 依序測試每個週期，直到找到含有 AP 資料的週期為止
    for target_cycle in all_cycles:
        try:
            target_dt = datetime.strptime(target_cycle, "%Y%m%d%H")
            # 檢查是否超過 24 小時 (24 小時前的週期就不抓了)
            if base_dt - target_dt > timedelta(hours=24):
                print(f"   ℹ️  [{typhoon_id}] 週期 {target_cycle}Z 已超過 24 小時限制，停止往前搜尋。")
                break
        except Exception:
            continue

        ap_data: dict[str, dict[int, dict]] = {}
        
        # 針對當前目標週期提取 AP 資料
        for line in lines: # 順正序或倒序皆可，因為我們有用 target_cycle 鎖定
            cols = [c.strip() for c in line.split(",")]
            if len(cols) < 10:
                continue
                
            current_cycle = cols[2]
            if current_cycle != target_cycle:
                continue
                
            tech = cols[4].upper()
            if not re.match(r'^AP\d{2}$', tech):
                continue
                
            try:
                tau      = int(cols[5])
                valid_dt = target_dt + timedelta(hours=tau)
                wind     = float(cols[8]) if cols[8].lstrip("-").isdigit() else None
                mslp     = float(cols[9]) if cols[9].lstrip("-").isdigit() else None
                if wind is None and mslp is None:
                    continue
                    
                ap_data.setdefault(tech, {}).setdefault(tau, {
                    "valid_time":  valid_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "coordinates": {"lat": _parse_latlon(cols[6]), "lon": _parse_latlon(cols[7])},
                    "intensity":   {"mslp_hpa": mslp, "max_wind_knots": wind},
                })
            except Exception:
                continue

        # 3. 如果這個週期成功撈到了 AP 資料，就直接重組回傳，不繼續往前找舊資料了
        if ap_data:
            print(f"   [解析] 成功鎖定 {typhoon_id} 的 AP 系集資料於週期: {target_cycle}Z")
            samples = [
                {
                    "sample_id":   float(int(re.sub(r'\D', '', m)) - 1),  # AP01→0.0
                    "data_points": [ap_data[m][tau] for tau in sorted(ap_data[m])],
                }
                for m in sorted(ap_data)
            ]
            return {typhoon_id: samples}
        else:
            print(f"   ⚠️  [{typhoon_id}] 週期 {target_cycle}Z 無 AP 系集資料，嘗試往前尋找...")

    # 如果所有 24 小時內的週期都找過了還是沒有
    return None

GFSE_JSON = "active_typhoon/cyclone_data_gfse.json"

def save_gfse_json(all_ap: dict) -> None:
    """將所有颱風的 AP 系集合併寫入單一 cyclone_data_gfse.json。"""
    os.makedirs("active_typhoon", exist_ok=True)
    with open(GFSE_JSON, "w", encoding="utf-8") as f:
        json.dump(all_ap, f, indent=4, ensure_ascii=False)
    print(f"✅ cyclone_data_gfse.json 儲存完畢（{len(all_ap)} 個風暴）")


# =============================================================================
# 主流程
# =============================================================================

def gfs_main():
    latest_date,  latest_cycle  = get_latest_available_cycle()
    prev_date,    prev_cycle    = get_previous_cycle(latest_date, latest_cycle)
    current_cid = cycle_id(latest_date, latest_cycle)

    print(f"最新週期: {current_cid}")
    print(f"前一週期: {cycle_id(prev_date, prev_cycle)}")
    print(f"輸出檔案: {OUTPUT_NPZ}")
    if current_cid in get_forecast_list():
        print(f"[跳過] {current_cid} 已於上次執行時處理完畢，無需重複下載。")
        return

    try:
        land_mask = np.load(LAND_MASK_PATH)
        print(f"陸地遮罩載入完成，形狀: {land_mask.shape}")
    except FileNotFoundError:
        print(f"[錯誤] 找不到陸地遮罩: {LAND_MASK_PATH}")
        return

    tasks = [
        {'date': prev_date,   'cycle': prev_cycle,   'hour': '000'},
        {'date': latest_date, 'cycle': latest_cycle, 'hour': '000'},
    ]
    paths = []
    for t in tasks:
        p = download_gfs(t['date'], t['cycle'], t['hour'])
        if p is None:
            print("[錯誤] 下載失敗，中止執行。")
            return
        paths.append(p)

    all_data, all_times = [], []
    channel_names = None

    for fp in paths:
        print(f"\n解析: {fp}")
        data, lats, lons, tv, names = parse_grib(fp)
        if data is None:
            print("[錯誤] 解析失敗，中止執行。")
            return
        if channel_names is None:
            channel_names = names
        all_data.append(data)
        all_times.append(tv)

    data = np.stack(all_data, axis=0)
    print(f"\n堆疊完成: {data.shape}")

    if data.shape[2] == 721:
        data = data[:, :, :-1, :]
        lats = lats[:-1]

    T, C, H, W  = data.shape
    nH = H // DOWNSAMPLE_FACTOR
    nW = W // DOWNSAMPLE_FACTOR
    data = (data
            .reshape(T, C, nH, DOWNSAMPLE_FACTOR, nW, DOWNSAMPLE_FACTOR)
            .mean(axis=(3, 5)))
    lats_ds = lats[DOWNSAMPLE_FACTOR // 2 :: DOWNSAMPLE_FACTOR]
    lons_ds = lons[DOWNSAMPLE_FACTOR // 2 :: DOWNSAMPLE_FACTOR]
    print(f"下採樣後: {data.shape}")

    for i, name in enumerate(channel_names):
        if name in FIXED_STATS:
            m, s = FIXED_STATS[name]['mean'], FIXED_STATS[name]['std']
            data[:, i] = (data[:, i] - m) / s
        else:
            print(f"[警告] {name} 未設定 FIXED_STATS，跳過標準化。")

    if 'sst' in channel_names:
        idx = channel_names.index('sst')
        if land_mask.shape == data.shape[2:]:
            for t in range(data.shape[0]):
                data[t, idx] = np.where(land_mask == 1, 0.0, data[t, idx])
            print("SST 陸地遮罩已套用。")
        else:
            print(f"[警告] 陸地遮罩形狀 {land_mask.shape} ≠ 資料 {data.shape[2:]}，跳過。")

    time_strs = np.array([
        np.datetime_as_string(tv, unit='h') for tv in all_times
    ])

    np.savez_compressed(
        OUTPUT_NPZ,
        data          = data.astype(np.float32),
        times         = time_strs,
        lats          = lats_ds.astype(np.float32),
        lons          = lons_ds.astype(np.float32),
        channel_names = np.array(channel_names),
    )
    print(f"\n✅ 已儲存 {OUTPUT_NPZ}  →  形狀 {data.shape}")


def google_main():
    os.makedirs("active_typhoon", exist_ok=True)
    try:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_tracks")
        os.makedirs(output_dir, exist_ok=True)

        for model_name in MODEL_CONFIGS:
            cfg = MODEL_CONFIGS[model_name]
            csv_path = download_model_data(model_name)
            convert_csv_to_json(csv_path, cfg["json_path"], model_name)
            os.remove(csv_path)
    except Exception as e:
        print(f"沒獲取到有效資料: {e}")

        
def active_main():
    """下載 24 小時內更新的 ATCF .dat 檔，儲存 active_list.json 與各颱風 AP 系集 JSON。"""
    urls = [
        "https://ftp.nhc.noaa.gov/atcf/jtwc/",
        "https://www.natyphoon.top/atcf/temp/"
    ]
    headers  = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    cutoff   = datetime.now() - timedelta(hours=24)

    active_list  : dict[str, dict]  = {}
    raw_contents : dict[str, bytes] = {}

    for base_url in urls:
        print(f"🔄 嘗試從網址獲取活躍颱風資料: {base_url}")
        try:
            response = requests.get(base_url, headers=headers, timeout=15)
            response.raise_for_status()
            
            html_text = response.text
            matches = []

            # 💡 判斷是否為 NTYSRV / NaTyphoon 類型的動態 JS 封裝網頁
            if "const FN =" in html_text and "const LM =" in html_text:
                print("   ℹ️ 偵測到動態 JS 封裝目錄，啟動 JS 變數解析器...")
                fn_match = re.search(r'const\s+FN\s*=\s*(\[.*?\]);', html_text, re.DOTALL)
                lm_match = re.search(r'const\s+LM\s*=\s*(\[.*?\]);', html_text, re.DOTALL)
                
                if fn_match and lm_match:
                    # 將 JS 陣列字串轉換為 Python 列表
                    fn_list = json.loads(fn_match.group(1))
                    lm_list = json.loads(lm_match.group(1))
                    
                    # 過濾並重新組合，使其符合 (dat_id, upd_time) 格式
                    for file_name, file_time in zip(fn_list, lm_list):
                        if re.match(r'^[ab][a-z0-9]+?\.dat$', file_name, re.I):
                            matches.append((file_name, file_time))
            else:
                # 💡 傳統官方靜態網頁結構：使用原先的 HTML 標籤正則掃描
                matches = re.findall(r'<a href="([ab][a-z0-9]+?\.dat)">.*?</a>\s+([\d:/-]+\s+[\d:]+)', html_text, re.I)

            if not matches:
                print(f"⚠️  該頁面未找到任何符合條件的 .dat 檔案。")
                continue

            print(f"   總共找到 {len(matches)} 個候選檔案，開始篩選 24 小時內更新的資料...")

            for dat_id, upd_time_raw in matches:
                try:
                    # 將時間字串中的正斜線统一替換為短橫線，確保格式統一
                    upd_time_str = upd_time_raw.replace('/', '-')
                    if datetime.strptime(upd_time_str, "%Y-%m-%d %H:%M") <= cutoff:
                        continue
                        
                    typhoon_id = dat_id[1:-4].upper()
                    resp       = requests.get(urljoin(base_url, dat_id), headers=headers, timeout=30)
                    resp.raise_for_status()
                    content    = resp.content

                    obs = parse_atcf_to_json(content, typhoon_id)
                    if obs:
                        active_list[typhoon_id] = obs

                    raw_contents[typhoon_id] = raw_contents.get(typhoon_id, b"") + b"\n" + content

                    save_dir = os.path.join("active_typhoon", typhoon_id)
                    os.makedirs(save_dir, exist_ok=True)
                    with open(os.path.join(save_dir, dat_id), "wb") as f:
                        f.write(content)
                except Exception as e:
                    continue

            # 如果當前網址成功找到並解析出活躍颱風，則直接中斷迴圈，不用再走後面的備用網址
            if active_list:
                print(f"✨ 成功於 {base_url} 獲取活躍颱風。")
                break
            else:
                print(f"ℹ️  網址 {base_url} 無符合 24 小時內更新的活動氣旋。")

        except Exception as e:
            print(f"❌ 請求 {base_url} 失敗: {e}")
            continue

    os.makedirs("active_typhoon", exist_ok=True)
    if active_list:
        with open("active_typhoon/active_list.json", "w", encoding="utf-8") as f:
            json.dump(active_list, f, ensure_ascii=False, indent=4)
        print(f"✅ active_list.json 儲存完畢（{len(active_list)} 個氣旋）")
    else:
        print("ℹ️  所有來源皆無符合 24 小時內更新的活動氣旋。")

    # ── AP 系集（合併所有風暴至單一 JSON）──────────────────────────────────
    print(f"\n{'='*60}\n📊 解析 AP 系集模型…")
    merged_ap = {}

    for typhoon_id, content in raw_contents.items():
        ap_result = parse_atcf_ap_ensemble(content, typhoon_id)
        if ap_result is None:
            print(f"  ⚠️  [{typhoon_id}] 無 AP 系集資料")
            continue
        merged_ap.update(ap_result)
        print(f"  ✅ [{typhoon_id}] {len(ap_result[typhoon_id])} 個成員")

    if merged_ap:
        save_gfse_json(merged_ap)
    else:
        print("ℹ️  無任何颱風含有 AP 系集資料。")

    print(f"{'='*60}")


if __name__ == "__main__":
    gfs_main()
    google_main()
    active_main()
