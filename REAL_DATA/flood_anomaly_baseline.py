import re
import os
import numpy as np
import rasterio
from datetime import datetime, date, timedelta
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple

# =========================================================
# USER SETTINGS
# =========================================================
base_tif_path  = "/Users/lunarolland/Desktop/DATASETS/Spain_7370579_S1_VV_VH_.tif"
extra_tif_path = "/Users/lunarolland/Desktop/DATASETS/Spain_7370579_S1_EXTRA_DESC_anyRelOrbit_20190911_20190929.tif"

# Outside focus window: use ONLY this platform from BASE
PLATFORM_BASE = "S1A"  # "S1A" or "S1B"

# Inside focus window: use BOTH platforms from EXTRA
PLATFORMS_EXTRA = ("S1A", "S1B")

# Flood center date (defines the 1-year before/after window)
flood_center_date = date(2019, 9, 17)

# Time window (score outputs for these dates)
start_date = flood_center_date - timedelta(days=365)
end_date   = flood_center_date + timedelta(days=365)

# Baseline window (pre-flood only)  [IMPORTANT: baseline uses BASE only]
ref_start = flood_center_date - timedelta(days=365)
ref_end   = flood_center_date - timedelta(days=1)

# Focus window: ONLY use EXTRA here (prefer extra if available)
focus_start = date(2019, 9, 11)
focus_end   = date(2019, 9, 29)

# Spatial smoothing scale (k=1 = none)
k = 5  # try 1,3,5,7,...

# Clamp ranges (dB)
VV_MIN, VV_MAX = -25.0, 0.0
VH_MIN, VH_MAX = -35.0, -5.0

# Flood score mix (VV usually stronger for open water)
W_VV, W_VH = 0.7, 0.3

# Baseline false positive rate target:
BASELINE_Q = 99.9

# Output folder
out_dir = "/Users/lunarolland/Desktop/flood_anomaly_outputs"
SAVE_SCORE_TIFS = True
SAVE_MASK_TIFS = True
SHOW_PLOTS = True

# Safer nodata for float GeoTIFFs than NaN
NODATA_F = -9999.0


# =========================================================
# HELPERS
# =========================================================
def parse_band(desc: str) -> Optional[dict]:
    """
    Parse band description to extract platform, timestamp, and polarization.

    Supports BOTH naming styles:
      - Old:  "S1A_20190917T053012_VV"
      - New:  "VV_20190917" or "VH_20190917" (from the nicer GEE renaming)
    """
    if not desc:
        return None

    d = desc.upper().strip()

    # polarization
    pol = None
    if d.startswith("VV_") or d.endswith("_VV"):
        pol = "VV"
    elif d.startswith("VH_") or d.endswith("_VH"):
        pol = "VH"

    # platform (may be missing in new style)
    plat = None
    if d.startswith("S1A_"):
        plat = "S1A"
    elif d.startswith("S1B_"):
        plat = "S1B"

    # datetime:
    # 1) old style with timestamp 20190917T053012
    m1 = re.search(r"(20\d{6}T\d{6})", d)
    if m1:
        dt = datetime.strptime(m1.group(1), "%Y%m%dT%H%M%S")
        return {"platform": plat, "pol": pol, "dt": dt, "desc": desc}

    # 2) new style with date only: VV_20190917
    m2 = re.search(r"(20\d{6})$", d)
    if m2 and (d.startswith("VV_") or d.startswith("VH_")):
        dt = datetime.strptime(m2.group(1), "%Y%m%d")
        return {"platform": plat, "pol": pol, "dt": dt, "desc": desc}

    return {"platform": plat, "pol": pol, "dt": None, "desc": desc}


def clamp(arr: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    a = arr.astype(np.float32)
    a = np.where(np.isfinite(a), a, np.nan)
    return np.clip(a, vmin, vmax)


def box_mean(arr: np.ndarray, k: int) -> np.ndarray:
    """Fast k×k box mean with reflect padding using integral image."""
    if k % 2 == 0:
        raise ValueError("k must be odd")
    if k == 1:
        return arr.astype(np.float32)

    pad = k // 2
    a = arr.astype(np.float32)
    valid = np.isfinite(a).astype(np.float32)
    a0 = np.where(np.isfinite(a), a, 0.0).astype(np.float32)

    a0p = np.pad(a0, pad, mode="reflect")
    vp  = np.pad(valid, pad, mode="reflect")

    S  = a0p.cumsum(0).cumsum(1)
    SV = vp.cumsum(0).cumsum(1)

    H, W = a.shape
    x2 = np.arange(k - 1, k - 1 + H)
    y2 = np.arange(k - 1, k - 1 + W)

    A = S[np.ix_(x2, y2)]
    B = S[np.ix_(x2 - k, y2)]
    C = S[np.ix_(x2, y2 - k)]
    D = S[np.ix_(x2 - k, y2 - k)]
    win_sum = A - B - C + D

    Av = SV[np.ix_(x2, y2)]
    Bv = SV[np.ix_(x2 - k, y2)]
    Cv = SV[np.ix_(x2, y2 - k)]
    Dv = SV[np.ix_(x2 - k, y2 - k)]
    win_cnt = Av - Bv - Cv + Dv

    out = win_sum / np.maximum(win_cnt, 1.0)
    out[win_cnt == 0] = np.nan
    return out.astype(np.float32)


def find_dates_available(bands: List[dict]) -> List[datetime]:
    return sorted({b["dt"] for b in bands if b["dt"] is not None})


def get_vh_vv_for_datetime(src, parsed_bands: List[dict], target_dt: datetime) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads VH and VV for target_dt.
    If multiple bands match (e.g., S1A + S1B), it averages them.
    """
    def read_band(bidx):
        arr = src.read(bidx).astype(np.float32)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        return arr

    out = {}
    for pol in ("VH", "VV"):
        matches = [b for b in parsed_bands if b["pol"] == pol and b["dt"] == target_dt]
        if not matches:
            raise RuntimeError(f"No {pol} band found for datetime {target_dt}.")
        arrs = [read_band(b["band"]) for b in matches]
        arr = np.nanmean(np.stack(arrs, axis=0), axis=0).astype(np.float32)

        if pol == "VV":
            arr = clamp(arr, VV_MIN, VV_MAX)
        else:
            arr = clamp(arr, VH_MIN, VH_MAX)

        out[pol] = arr
    return out["VH"], out["VV"]


def robust_baseline(stack: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """stack: (T,H,W) returns: median (H,W), IQR (H,W)"""
    med = np.nanmedian(stack, axis=0)
    q25 = np.nanpercentile(stack, 25, axis=0)
    q75 = np.nanpercentile(stack, 75, axis=0)
    iqr = q75 - q25
    return med.astype(np.float32), iqr.astype(np.float32)


def flood_score(VH_t, VV_t, VH_med, VH_iqr, VV_med, VV_iqr, eps=1e-3):
    z_vv = (VV_t - VV_med) / (VV_iqr + eps)
    z_vh = (VH_t - VH_med) / (VH_iqr + eps)
    score_vv = -z_vv
    score_vh = -z_vh
    score = W_VV * score_vv + W_VH * score_vh
    return score.astype(np.float32), score_vv.astype(np.float32), score_vh.astype(np.float32)


def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def write_geotiff(path: str, arr: np.ndarray, profile: dict, dtype, nodata=None):
    prof = profile.copy()
    prof.update(count=1, dtype=dtype, nodata=nodata)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr.astype(dtype), 1)


def in_focus_window(d: date) -> bool:
    return (focus_start <= d <= focus_end)


# =========================================================
# MAIN
# =========================================================
def main():
    safe_mkdir(out_dir)

    # Open both rasters
    with rasterio.open(base_tif_path) as base_src, rasterio.open(extra_tif_path) as extra_src:
        # Safety: require same grid
        if (base_src.width != extra_src.width or base_src.height != extra_src.height or
            base_src.transform != extra_src.transform or base_src.crs != extra_src.crs):
            raise RuntimeError("BASE and EXTRA GeoTIFFs are not on the same grid/CRS/transform. "
                               "Export them with identical region/scale/crs so they align.")

        profile = base_src.profile

        # ---- Parse BASE bands ----
        base_parsed = []
        for band_i, desc in enumerate(base_src.descriptions, start=1):
            info = parse_band(desc)
            if info is None:
                continue
            info["band"] = band_i
            base_parsed.append(info)

        # Keep only desired platform for BASE (outside focus)
        base_parsed = [b for b in base_parsed
                       if b["pol"] in ("VV", "VH") and b["dt"] is not None and b["platform"] == PLATFORM_BASE]

        if not base_parsed:
            raise RuntimeError(f"No BASE bands found for platform {PLATFORM_BASE} with parsable dates.")

        # ---- Parse EXTRA bands ----
        extra_parsed = []
        for band_i, desc in enumerate(extra_src.descriptions, start=1):
            info = parse_band(desc)
            if info is None:
                continue
            info["band"] = band_i
            extra_parsed.append(info)

        # For EXTRA: allow both platforms (even if platform missing in band names, we still accept)
        # If platform is None (new naming), we accept it.
        extra_parsed = [b for b in extra_parsed
                        if b["pol"] in ("VV", "VH") and b["dt"] is not None and
                        (b["platform"] in PLATFORMS_EXTRA or b["platform"] is None)]

        # ---- Build available date lists ----
        base_dts = find_dates_available(base_parsed)
        extra_dts = find_dates_available(extra_parsed)

        # Overall scoring dates:
        # - outside focus: use BASE dates
        # - inside focus: use EXTRA dates (prefer), but we still keep BASE dates outside focus
        base_dts_window = [dt for dt in base_dts if start_date <= dt.date() <= end_date and not in_focus_window(dt.date())]
        extra_dts_window = [dt for dt in extra_dts if start_date <= dt.date() <= end_date and in_focus_window(dt.date())]

        # final timeline is base(outside focus) + extra(inside focus)
        dts_window = sorted(base_dts_window + extra_dts_window)

        # Baseline dates (pre-flood only) from BASE only (outside focus by definition)
        dts_ref = [dt for dt in base_dts if ref_start <= dt.date() <= ref_end]

        print(f"BASE acquisitions total ({PLATFORM_BASE}): {len(base_dts)}")
        print(f"EXTRA acquisitions total (focus stack): {len(extra_dts)}")
        print(f"Window acquisitions total (BASE outside focus + EXTRA inside focus): {len(dts_window)}")
        print(f"  - BASE outside-focus dates: {len(base_dts_window)}")
        print(f"  - EXTRA focus dates:       {len(extra_dts_window)}")
        print(f"Baseline acquisitions (BASE only) ({ref_start}..{ref_end}): {len(dts_ref)}")

        if len(dts_ref) < 5:
            print("WARNING: Very few baseline dates. Baseline may be unstable.")

        # ----- Build baseline stacks (BASE only) -----
        print(f"\nBuilding per-pixel baseline using k={k} (BASE only) ...")
        VV_stack = []
        VH_stack = []
        for dt in dts_ref:
            VH, VV = get_vh_vv_for_datetime(base_src, base_parsed, dt)
            if k > 1:
                VH = box_mean(VH, k)
                VV = box_mean(VV, k)
            VV_stack.append(VV)
            VH_stack.append(VH)

        VV_stack = np.stack(VV_stack, axis=0)
        VH_stack = np.stack(VH_stack, axis=0)

        VV_med, VV_iqr = robust_baseline(VV_stack)
        VH_med, VH_iqr = robust_baseline(VH_stack)

        # ----- Compute baseline threshold -----
        print(f"\nCalibrating global threshold from baseline at {BASELINE_Q}th percentile ...")
        baseline_scores = []
        for i, dt in enumerate(dts_ref):
            VH_t = VH_stack[i]
            VV_t = VV_stack[i]
            score, _, _ = flood_score(VH_t, VV_t, VH_med, VH_iqr, VV_med, VV_iqr)
            baseline_scores.append(score[np.isfinite(score)])

        baseline_scores = np.concatenate(baseline_scores) if baseline_scores else np.array([], dtype=np.float32)
        if baseline_scores.size == 0:
            raise RuntimeError("No finite baseline scores; check nodata handling / clamps.")

        THR_SCORE = float(np.percentile(baseline_scores, BASELINE_Q))
        print(f"THR_SCORE (from BASE baseline) = {THR_SCORE:.3f}\n")

        # ----- Global plot scaling across the FULL mixed timeline -----
        if SHOW_PLOTS:
            print("Computing global plot scaling for score (consistent across dates)...")
            all_scores = []
            for dt in dts_window:
                # choose source by date
                if in_focus_window(dt.date()):
                    src_use, parsed_use = extra_src, extra_parsed
                else:
                    src_use, parsed_use = base_src, base_parsed

                VH_t, VV_t = get_vh_vv_for_datetime(src_use, parsed_use, dt)
                if k > 1:
                    VH_t = box_mean(VH_t, k)
                    VV_t = box_mean(VV_t, k)
                score, _, _ = flood_score(VH_t, VV_t, VH_med, VH_iqr, VV_med, VV_iqr)
                all_scores.append(score)

            S = np.stack(all_scores, axis=0)
            PLOT_VMIN = float(np.nanpercentile(S, 2))
            PLOT_VMAX = float(np.nanpercentile(S, 98))
            print(f"PLOT_VMIN={PLOT_VMIN:.3f}, PLOT_VMAX={PLOT_VMAX:.3f}\n")

        # ----- Score each date -----
        print("Scoring dates and writing outputs...")
        for dt in dts_window:
            use_extra = in_focus_window(dt.date())
            src_use, parsed_use = (extra_src, extra_parsed) if use_extra else (base_src, base_parsed)

            VH_t, VV_t = get_vh_vv_for_datetime(src_use, parsed_use, dt)
            if k > 1:
                VH_t = box_mean(VH_t, k)
                VV_t = box_mean(VV_t, k)

            score, _, _ = flood_score(VH_t, VV_t, VH_med, VH_iqr, VV_med, VV_iqr)
            mask = (score >= THR_SCORE).astype(np.uint8)

            finite = np.isfinite(score)
            frac = float(mask[finite].mean()) if finite.any() else 0.0
            tag = dt.strftime("%Y%m%dT%H%M%S")
            src_tag = "EXTRA" if use_extra else "BASE"
            print(f"{dt.date()} [{src_tag}] flooded_fraction={frac*100:.3f}%")

            if SAVE_SCORE_TIFS:
                score_out = np.where(np.isfinite(score), score, NODATA_F).astype(np.float32)
                write_geotiff(
                    os.path.join(out_dir, f"score_{src_tag}_{tag}_k{k}.tif"),
                    score_out, profile, dtype="float32", nodata=NODATA_F
                )

            if SAVE_MASK_TIFS:
                write_geotiff(
                    os.path.join(out_dir, f"mask_{src_tag}_{tag}_k{k}.tif"),
                    mask, profile, dtype="uint8", nodata=0
                )

            if SHOW_PLOTS:
                plt.figure(figsize=(12, 4))
                plt.subplot(1, 3, 1)
                plt.imshow(score, cmap="magma", vmin=PLOT_VMIN, vmax=PLOT_VMAX)
                plt.title(f"Flood score ({src_tag})\n{dt.date()}")
                plt.axis("off")

                plt.subplot(1, 3, 2)
                plt.imshow(mask, cmap="gray", vmin=0, vmax=1)
                plt.title("Flood mask")
                plt.axis("off")

                plt.subplot(1, 3, 3)
                plt.imshow(VV_t, vmin=VV_MIN, vmax=VV_MAX, cmap="gray")
                plt.title("VV (processed)")
                plt.axis("off")

                plt.tight_layout()
                plt.show()

        # ----- Time series plots -----
        print("\nBuilding flood intensity time series...")
        dates_ts = []
        mean_top1_scores = []
        flooded_fraction_ts = []
        src_used_ts = []

        for dt in dts_window:
            use_extra = in_focus_window(dt.date())
            src_use, parsed_use = (extra_src, extra_parsed) if use_extra else (base_src, base_parsed)

            VH_t, VV_t = get_vh_vv_for_datetime(src_use, parsed_use, dt)
            if k > 1:
                VH_t = box_mean(VH_t, k)
                VV_t = box_mean(VV_t, k)

            score, _, _ = flood_score(VH_t, VV_t, VH_med, VH_iqr, VV_med, VV_iqr)
            finite = score[np.isfinite(score)]
            if finite.size == 0:
                continue

            thr_top1 = np.percentile(finite, 99)
            mean_top1 = float(finite[finite >= thr_top1].mean())

            mask = (score >= THR_SCORE)
            frac = float(mask[np.isfinite(score)].mean()) * 100.0

            dates_ts.append(dt.date())
            mean_top1_scores.append(mean_top1)
            flooded_fraction_ts.append(frac)
            src_used_ts.append("EXTRA" if use_extra else "BASE")

        if SHOW_PLOTS and len(dates_ts) > 0:
            plt.figure(figsize=(10, 5))
            plt.plot(dates_ts, mean_top1_scores, marker='o')
            plt.axvline(flood_center_date, linestyle='--')
            plt.title("Flood Intensity (Mean Top 1% Score)")
            plt.ylabel("Mean Top 1% Score")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(10, 5))
            plt.plot(dates_ts, flooded_fraction_ts, marker='o')
            plt.axvline(flood_center_date, linestyle='--')
            plt.title("Flooded Area Fraction (%)")
            plt.ylabel("Flooded Pixels (%)")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

    print("\nDone. Load mask_*.tif in QGIS to see flooded pixels and compute area stats.")


if __name__ == "__main__":
    main()
