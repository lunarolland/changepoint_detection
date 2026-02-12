"""
Flood Detection Pipeline for Sentinel-1 SAR Data  (numeric-change focused)

UPDATED to match what you asked:
- Uses S1A instead of S1B
- Compares TWO target dates:
  * before_target_date (e.g. 2019-09-03): picks last acquisition on/before target (else closest)
  * during_target_date (e.g. 2019-09-17): picks first acquisition on/after target (else closest)

Everything else is kept the same unless it MUST change for the above.
"""

import re
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from datetime import datetime, date
from typing import Tuple, List, Optional
import pandas as pd


# ============================================================================
# USER SETTINGS
# ============================================================================
tif_path = "/Users/lunarolland/Desktop/DATASETS/Spain_7370579_S1_VV_VH_.tif"

# Platform switch (YOU ASKED: use S1A instead of S1B)
PLATFORM = "S1A"   # "S1A" or "S1B"

# Your two comparison targets
before_target_date = date(2019, 9, 3)
during_target_date = date(2019, 9, 17)

# Box size (odd): k=5 means 5x5 neighborhood averaging around each pixel
k = 5

# Clamp ranges (dB) - values outside these ranges will be clipped
VV_MIN, VV_MAX = -25.0, 0.0
VH_MIN, VH_MAX = -35.0, -5.0

# Visualization settings (optional)
SHOW_PLOTS = True  # True/False

# Quantile percentages for numeric summaries
Q_PCTS = [1, 5, 25, 50, 75, 95, 99]

# To show "changed a lot" vs "changed a little" WITHOUT a chosen threshold:
# we use quantile tails of the change image (e.g., bottom 10% vs top 10%).
TAIL_PCT = 10  # 10 means bottom 10% vs top 10%

# Export
EXPORT_SUMMARY_CSV = True
summary_csv_path = "/Users/lunarolland/Desktop/flood_numeric_summary.csv"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def parse_band(desc: str) -> Optional[dict]:
    """Parse band description to extract platform, timestamp, and polarization."""
    if desc is None:
        return None

    d = desc.upper()

    plat = "S1B" if d.startswith("S1B_") else ("S1A" if d.startswith("S1A_") else None)
    pol = "VV" if d.endswith("_VV") else ("VH" if d.endswith("_VH") else None)

    m = re.search(r"(20\d{6}T\d{6})", d)
    dt = datetime.strptime(m.group(1), "%Y%m%dT%H%M%S") if m else None

    return {"platform": plat, "pol": pol, "dt": dt, "desc": desc}


def clamp(arr: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Clamp array values to [vmin, vmax], preserving NaNs."""
    a = arr.astype(np.float32)
    a = np.where(np.isfinite(a), a, np.nan)
    return np.clip(a, vmin, vmax)


def box_mean(arr: np.ndarray, k: int) -> np.ndarray:
    """Fast k×k box mean (spatial averaging) with reflect padding using integral image."""
    if k % 2 == 0:
        raise ValueError("k must be odd")
    pad = k // 2

    a = arr.astype(np.float32)
    valid = np.isfinite(a).astype(np.float32)
    a0 = np.where(np.isfinite(a), a, 0.0).astype(np.float32)

    a0p = np.pad(a0, pad, mode="reflect")
    vp = np.pad(valid, pad, mode="reflect")

    S = a0p.cumsum(0).cumsum(1)
    SV = vp.cumsum(0).cumsum(1)

    H, W = arr.shape
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


def stack_vh_vv(VH: np.ndarray, VV: np.ndarray) -> np.ndarray:
    """Stack VH and VV arrays into shape (H, W, 2) with [:,:,0]=VH, [:,:,1]=VV."""
    return np.stack([VH, VV], axis=-1).astype(np.float32)


def find_dates_available(bands: List[dict]) -> List[datetime]:
    """Extract sorted unique acquisition datetimes from parsed bands."""
    return sorted({b["dt"] for b in bands if b["dt"] is not None})


def pick_before_dt(dts: List[datetime], target: date) -> datetime:
    """Prefer last acquisition on/before target; else closest by absolute days."""
    if not dts:
        raise RuntimeError("No datetimes available.")
    on_or_before = [dt for dt in dts if dt.date() <= target]
    if on_or_before:
        return max(on_or_before, key=lambda dt: dt.date())
    return min(dts, key=lambda dt: abs((dt.date() - target).days))


def pick_during_dt(dts: List[datetime], target: date) -> datetime:
    """Prefer first acquisition on/after target; else closest by absolute days."""
    if not dts:
        raise RuntimeError("No datetimes available.")
    on_or_after = [dt for dt in dts if dt.date() >= target]
    if on_or_after:
        return min(on_or_after, key=lambda dt: dt.date())
    return min(dts, key=lambda dt: abs((dt.date() - target).days))


def get_vh_vv_for_datetime(src, parsed_bands: List[dict], target_dt: datetime) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract clamped VH and VV arrays for a given datetime.
    NOTE: platform filtering is done upstream; here we only match dt+pol.
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


# ============================================================================
# NUMERIC SUMMARY HELPERS (NO THRESHOLDS)
# ============================================================================

def _finite(a: np.ndarray) -> np.ndarray:
    return a[np.isfinite(a)]


def stats_row(arr: np.ndarray, name: str, layer: str, dt_label: str) -> dict:
    """Return mean/std + quantiles for finite values."""
    a = _finite(arr)
    row = {"layer": layer, "name": name, "dt": dt_label, "n": int(a.size)}
    if a.size == 0:
        return row
    qs = np.percentile(a, Q_PCTS)
    row["mean"] = float(a.mean())
    row["std"] = float(a.std())
    for p, v in zip(Q_PCTS, qs):
        row[f"p{p:02d}"] = float(v)
    return row


def quantile_slice_stats(change_arr: np.ndarray, tail_pct: float, name: str, dt_label: str) -> List[dict]:
    """
    Show numeric contrast between:
      - bottom tail_pct% of change (strongest negative change)
      - top tail_pct% of change (weak/positive change)
    This is NOT a hand-picked threshold; it's an automatic quantile split.
    """
    a = change_arr
    af = _finite(a)
    if af.size == 0:
        return [
            {"layer": "change_slice", "name": f"{name}_bottom{tail_pct}%", "dt": dt_label, "n": 0},
            {"layer": "change_slice", "name": f"{name}_top{tail_pct}%", "dt": dt_label, "n": 0},
        ]

    lo = np.percentile(af, tail_pct)
    hi = np.percentile(af, 100 - tail_pct)

    bottom = a[(np.isfinite(a)) & (a <= lo)]
    top = a[(np.isfinite(a)) & (a >= hi)]

    return [
        stats_row(bottom, f"{name}_bottom{tail_pct}%", "change_slice", dt_label),
        stats_row(top,    f"{name}_top{tail_pct}%",    "change_slice", dt_label),
    ]


def print_df(df: pd.DataFrame, title: str):
    print("\n" + title)
    print("-" * len(title))
    with pd.option_context("display.max_columns", None, "display.width", 180):
        print(df)


# ============================================================================
# MAIN PROCESSING PIPELINE
# ============================================================================

def process_flood_data(tif_path: str):
    """
    Returns:
      M_raw_before, M_raw_during, M_box_before, M_box_during, dts, before_dt, during_dt
    """
    with rasterio.open(tif_path) as src:
        descs = list(src.descriptions)
        if not descs or all(d is None for d in descs):
            raise RuntimeError("This GeoTIFF has no band descriptions; cannot parse S1A/S1B/date/pol.")

        parsed = []
        for band_i, desc in enumerate(descs, start=1):
            info = parse_band(desc)
            if info is None:
                continue
            info["band"] = band_i
            parsed.append(info)

        # Filter for chosen PLATFORM VV/VH only
        parsed_plat = [
            b for b in parsed
            if b["platform"] == PLATFORM and b["pol"] in ("VV", "VH") and b["dt"] is not None
        ]
        if not parsed_plat:
            raise RuntimeError(f"No {PLATFORM} VV/VH bands found in this GeoTIFF.")

        dts = find_dates_available(parsed_plat)
        print(f"Found {len(dts)} unique {PLATFORM} acquisition datetimes")

        # Pick your two comparison acquisitions
        before_dt = pick_before_dt(dts, before_target_date)
        during_dt = pick_during_dt(dts, during_target_date)

        print(f"Selected BEFORE datetime: {before_dt} (date: {before_dt.date()})  target={before_target_date}")
        print(f"Selected DURING datetime: {during_dt} (date: {during_dt.date()})  target={during_target_date}")

        print("Extracting VH/VV data...")
        VH_before, VV_before = get_vh_vv_for_datetime(src, parsed_plat, before_dt)
        VH_during, VV_during = get_vh_vv_for_datetime(src, parsed_plat, during_dt)

        M_raw_before = stack_vh_vv(VH_before, VV_before)
        M_raw_during = stack_vh_vv(VH_during, VV_during)

        print(f"Applying {k}×{k} box-averaging...")
        VH_before_box = box_mean(VH_before, k)
        VV_before_box = box_mean(VV_before, k)
        VH_during_box = box_mean(VH_during, k)
        VV_during_box = box_mean(VV_during, k)

        M_box_before = stack_vh_vv(VH_before_box, VV_before_box)
        M_box_during = stack_vh_vv(VH_during_box, VV_during_box)

        print(f"\nMatrix shapes:")
        print(f"  M_raw_before: {M_raw_before.shape} (dtype: {M_raw_before.dtype})")
        print(f"  M_raw_during: {M_raw_during.shape} (dtype: {M_raw_during.dtype})")
        print(f"  M_box_before: {M_box_before.shape} (dtype: {M_box_before.dtype})")
        print(f"  M_box_during: {M_box_during.shape} (dtype: {M_box_during.dtype})")

        return M_raw_before, M_raw_during, M_box_before, M_box_during, dts, before_dt, during_dt


def visualize_flood_comparison(M_raw_before, M_raw_during, M_box_before, M_box_during,
                               before_dt: datetime, during_dt: datetime):
    """Optional: show the same plots as before (kept)."""
    VH_raw_before = M_raw_before[:, :, 0]
    VV_raw_before = M_raw_before[:, :, 1]
    VH_raw_during = M_raw_during[:, :, 0]
    VV_raw_during = M_raw_during[:, :, 1]

    VH_box_before = M_box_before[:, :, 0]
    VV_box_before = M_box_before[:, :, 1]
    VH_box_during = M_box_during[:, :, 0]
    VV_box_during = M_box_during[:, :, 1]

    fig1, axs1 = plt.subplots(2, 2, figsize=(14, 12))

    im1 = axs1[0, 0].imshow(VH_raw_before, cmap="gray", vmin=VH_MIN, vmax=VH_MAX)
    axs1[0, 0].set_title(f"VH Raw - Before\n{before_dt.date()}", fontsize=12, fontweight="bold")
    axs1[0, 0].axis("off")
    plt.colorbar(im1, ax=axs1[0, 0], label="dB")

    im2 = axs1[0, 1].imshow(VH_raw_during, cmap="gray", vmin=VH_MIN, vmax=VH_MAX)
    axs1[0, 1].set_title(f"VH Raw - During\n{during_dt.date()}", fontsize=12, fontweight="bold")
    axs1[0, 1].axis("off")
    plt.colorbar(im2, ax=axs1[0, 1], label="dB")

    im3 = axs1[1, 0].imshow(VV_raw_before, cmap="gray", vmin=VV_MIN, vmax=VV_MAX)
    axs1[1, 0].set_title(f"VV Raw - Before\n{before_dt.date()}", fontsize=12, fontweight="bold")
    axs1[1, 0].axis("off")
    plt.colorbar(im3, ax=axs1[1, 0], label="dB")

    im4 = axs1[1, 1].imshow(VV_raw_during, cmap="gray", vmin=VV_MIN, vmax=VV_MAX)
    axs1[1, 1].set_title(f"VV Raw - During\n{during_dt.date()}", fontsize=12, fontweight="bold")
    axs1[1, 1].axis("off")
    plt.colorbar(im4, ax=axs1[1, 1], label="dB")

    plt.suptitle("Raw Pixel Values: Before vs During", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()

    fig2, axs2 = plt.subplots(2, 2, figsize=(14, 12))

    im5 = axs2[0, 0].imshow(VH_box_before, cmap="gray", vmin=VH_MIN, vmax=VH_MAX)
    axs2[0, 0].set_title(f"VH Box ({k}×{k}) - Before\n{before_dt.date()}", fontsize=12, fontweight="bold")
    axs2[0, 0].axis("off")
    plt.colorbar(im5, ax=axs2[0, 0], label="dB")

    im6 = axs2[0, 1].imshow(VH_box_during, cmap="gray", vmin=VH_MIN, vmax=VH_MAX)
    axs2[0, 1].set_title(f"VH Box ({k}×{k}) - During\n{during_dt.date()}", fontsize=12, fontweight="bold")
    axs2[0, 1].axis("off")
    plt.colorbar(im6, ax=axs2[0, 1], label="dB")

    im7 = axs2[1, 0].imshow(VV_box_before, cmap="gray", vmin=VV_MIN, vmax=VV_MAX)
    axs2[1, 0].set_title(f"VV Box ({k}×{k}) - Before\n{before_dt.date()}", fontsize=12, fontweight="bold")
    axs2[1, 0].axis("off")
    plt.colorbar(im7, ax=axs2[1, 0], label="dB")

    im8 = axs2[1, 1].imshow(VV_box_during, cmap="gray", vmin=VV_MIN, vmax=VV_MAX)
    axs2[1, 1].set_title(f"VV Box ({k}×{k}) - During\n{during_dt.date()}", fontsize=12, fontweight="bold")
    axs2[1, 1].axis("off")
    plt.colorbar(im8, ax=axs2[1, 1], label="dB")

    plt.suptitle(f"Box-Averaged ({k}×{k}): Before vs During", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.show()


# ============================================================================
# ADD-ON: FIXED dB BIN ("6 categories") TIME SERIES FOR MULTIPLE k
# (This is NOT quantiles; it's fixed-value bin percentages over time.)
# ============================================================================

from datetime import timedelta

# ---- USER SETTINGS FOR BIN TIME SERIES ----
RUN_BIN_TIMESERIES = True

# Define flood "center date" (used only to define your 1-year before/after window)
flood_center_date = date(2019, 9, 11)  # edit if needed

# Time window: 1 year before to 1 year after
ts_start_date = flood_center_date - timedelta(days=365)
ts_end_date   = flood_center_date + timedelta(days=365)

# k values to test (include 1 = no box averaging)
k_list = [1, 3, 5, 7, 9]

# Fixed dB bins (your idea)
# VH is clamped to [-35, -5] in your script
VH_BINS = [-35, -30, -25, -20, -15, -10, -5]  # 6 bins

# VV is clamped to [-25, 0] in your script
# To get 6 bins neatly: choose step=4 dB with a small remainder handled by last bin
VV_BINS = [-25, -21, -17, -13, -9, -5, 0]     # 6 bins

EXPORT_BIN_TS_CSV = True
bin_ts_csv_path = "/Users/lunarolland/Desktop/flood_bin_timeseries.csv"


def bin_percentages(arr: np.ndarray, bin_edges: List[float]) -> np.ndarray:
    """
    Returns percentage of finite pixels in each fixed-value bin.
    bins: [e0,e1), [e1,e2), ... [e_{n-1}, e_n)
    """
    a = arr[np.isfinite(arr)]
    if a.size == 0:
        return np.zeros(len(bin_edges) - 1, dtype=np.float32)
    counts, _ = np.histogram(a, bins=np.array(bin_edges, dtype=np.float32))
    return (counts / a.size * 100.0).astype(np.float32)


def build_binned_timeseries(tif_path: str,
                            k_list: List[int],
                            start_date: date,
                            end_date: date,
                            vh_bins: List[float],
                            vv_bins: List[float]) -> pd.DataFrame:
    """
    For each acquisition datetime within [start_date, end_date], and each k:
      - read VH/VV
      - apply box_mean if k>1
      - compute % of pixels in each fixed dB bin
    Returns a tidy DataFrame with columns:
      date, dt, k, pol, bin, pct, n_valid
    """
    rows = []

    with rasterio.open(tif_path) as src:
        descs = list(src.descriptions)
        if not descs or all(d is None for d in descs):
            raise RuntimeError("This GeoTIFF has no band descriptions; cannot parse platform/date/pol.")

        parsed = []
        for band_i, desc in enumerate(descs, start=1):
            info = parse_band(desc)
            if info is None:
                continue
            info["band"] = band_i
            parsed.append(info)

        parsed_plat = [
            b for b in parsed
            if b["platform"] == PLATFORM and b["pol"] in ("VV", "VH") and b["dt"] is not None
        ]
        if not parsed_plat:
            raise RuntimeError(f"No {PLATFORM} VV/VH bands found for this GeoTIFF.")

        dts_all = find_dates_available(parsed_plat)
        dts = [dt for dt in dts_all if start_date <= dt.date() <= end_date]
        dts = sorted(dts)

        print(f"\nBIN-TS: Found {len(dts)} acquisition datetimes in window {start_date} .. {end_date}")

        for k0 in k_list:
            if k0 % 2 == 0:
                raise ValueError(f"k must be odd; got {k0}")

            print(f"\nBIN-TS: Processing k={k0} ...")
            for dt in dts:
                VH, VV = get_vh_vv_for_datetime(src, parsed_plat, dt)

                if k0 > 1:
                    VH = box_mean(VH, k0)
                    VV = box_mean(VV, k0)

                # Count valid pixels (same definition as your stats helpers)
                n_vh = int(np.isfinite(VH).sum())
                n_vv = int(np.isfinite(VV).sum())

                vh_pct = bin_percentages(VH, vh_bins)
                vv_pct = bin_percentages(VV, vv_bins)

                # Store per-bin rows
                for i in range(len(vh_bins) - 1):
                    rows.append({
                        "date": dt.date(),
                        "dt": dt,
                        "k": k0,
                        "pol": "VH",
                        "bin": f"[{vh_bins[i]},{vh_bins[i+1]})",
                        "pct": float(vh_pct[i]),
                        "n_valid": n_vh
                    })

                for i in range(len(vv_bins) - 1):
                    rows.append({
                        "date": dt.date(),
                        "dt": dt,
                        "k": k0,
                        "pol": "VV",
                        "bin": f"[{vv_bins[i]},{vv_bins[i+1]})",
                        "pct": float(vv_pct[i]),
                        "n_valid": n_vv
                    })

    df = pd.DataFrame(rows)
    return df


def plot_bins_over_time(df: pd.DataFrame, pol: str = "VV"):
    """
    For each k, plot 6 lines (bins) of pct vs date.
    """
    dff = df[df["pol"] == pol].copy()
    if dff.empty:
        print(f"No rows to plot for pol={pol}")
        return

    # Sort bins by their numeric lower edge, not string order
    def bin_key(b: str) -> float:
        # b like "[-35,-30)"
        try:
            return float(b.split(",")[0].strip()[1:])
        except Exception:
            return 0.0

    for k0 in sorted(dff["k"].unique()):
        sub = dff[dff["k"] == k0].copy()
        sub = sub.sort_values("date")

        plt.figure(figsize=(12, 5))
        for b in sorted(sub["bin"].unique(), key=bin_key):
            s2 = sub[sub["bin"] == b]
            plt.plot(s2["date"], s2["pct"], label=b)

        plt.title(f"{pol}: % pixels in fixed dB bins over time (k={k0})")
        plt.xlabel("Date")
        plt.ylabel("% of valid pixels")
        plt.legend(title="Bin (dB)", ncols=3, fontsize=9)
        plt.tight_layout()
        plt.show()


# ============================================================================
# NEW ADD-ON PLOTS YOU ASKED FOR (VV-only “isolated bin(s)” plots)
# ============================================================================

def plot_vv_single_bin(df: pd.DataFrame, bin_label: str, k_values=None):
    """
    Plot ONLY one VV bin across time.
    If k_values is None -> plots for all k in df.
    """
    dff = df[(df["pol"] == "VV") & (df["bin"] == bin_label)].copy()
    if dff.empty:
        print(f"No rows found for VV bin={bin_label}")
        return

    ks = sorted(dff["k"].unique()) if k_values is None else sorted(set(k_values))
    for k0 in ks:
        s2 = dff[dff["k"] == k0].sort_values("date")
        if s2.empty:
            continue
        plt.figure(figsize=(12, 4))
        plt.plot(s2["date"], s2["pct"], label=bin_label)
        plt.title(f"VV: {bin_label} only (k={k0})")
        plt.xlabel("Date")
        plt.ylabel("% of valid pixels")
        plt.legend()
        plt.tight_layout()
        plt.show()


def plot_vv_two_bins(df: pd.DataFrame, bin_a: str, bin_b: str, k_values=None, also_plot_sum=True):
    """
    Plot ONLY two VV bins across time.
    Optionally also plot their SUM as a single curve (useful if you want “blue+orange combined”).
    """
    dff = df[(df["pol"] == "VV") & (df["bin"].isin([bin_a, bin_b]))].copy()
    if dff.empty:
        print(f"No rows found for VV bins={bin_a}, {bin_b}")
        return

    ks = sorted(dff["k"].unique()) if k_values is None else sorted(set(k_values))
    for k0 in ks:
        sub = dff[dff["k"] == k0].sort_values("date")
        if sub.empty:
            continue

        plt.figure(figsize=(12, 4))

        for b in [bin_a, bin_b]:
            s2 = sub[sub["bin"] == b].sort_values("date")
            if not s2.empty:
                plt.plot(s2["date"], s2["pct"], label=b)

        if also_plot_sum:
            wide = sub.pivot_table(index="date", columns="bin", values="pct", aggfunc="mean")
            if (bin_a in wide.columns) and (bin_b in wide.columns):
                sum_series = (wide[bin_a] + wide[bin_b]).sort_index()
                plt.plot(sum_series.index, sum_series.values, label=f"{bin_a}+{bin_b} (sum)")

        plt.title(f"VV: {bin_a} and {bin_b} (k={k0})")
        plt.xlabel("Date")
        plt.ylabel("% of valid pixels")
        plt.legend()
        plt.tight_layout()
        plt.show()


# ============================================================================
# ADD-ON METHOD 2: HISTOGRAM DISTANCE TO PRE-FLOOD BASELINE (ONE SCORE CURVE)
# - Uses your fixed dB bins
# - Builds a baseline histogram from 1-year pre-flood
# - For each date: histogram -> distance from baseline -> score
# - Plot score vs time (cleaner flood detection signal)
# ============================================================================

RUN_HIST_DISTANCE = True

EXPORT_HIST_SCORE_CSV = True
hist_score_csv_path = "/Users/lunarolland/Desktop/flood_hist_distance_scores.csv"

def hist_prob(arr: np.ndarray, bin_edges: List[float]) -> np.ndarray:
    """Normalized histogram (probabilities) for finite pixels."""
    a = arr[np.isfinite(arr)]
    if a.size == 0:
        return np.zeros(len(bin_edges) - 1, dtype=np.float32)
    counts, _ = np.histogram(a, bins=np.array(bin_edges, dtype=np.float32))
    p = counts.astype(np.float32)
    s = p.sum()
    return p / s if s > 0 else p

def l1_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Simple distance: sum |p-q|."""
    return float(np.sum(np.abs(p - q)))

def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """
    Jensen–Shannon divergence (stable, bounded-ish). Higher = more different.
    Uses natural log.
    """
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5*np.sum(p*np.log(p/m)) + 0.5*np.sum(q*np.log(q/m)))

def build_hist_distance_timeseries(tif_path: str,
                                   k_list: List[int],
                                   start_date: date,
                                   end_date: date,
                                   flood_date: date,
                                   vv_bins: List[float],
                                   vh_bins: List[float],
                                   use_js: bool = True) -> pd.DataFrame:
    """
    Returns tidy DataFrame: date, k, pol, score
    where score = distance(hist_t, hist_baseline_preFlood)
    """
    # Baseline window: 1 year pre-flood
    ref_start = flood_date - timedelta(days=365)
    ref_end   = flood_date - timedelta(days=1)

    rows = []

    with rasterio.open(tif_path) as src:
        # parse bands
        descs = list(src.descriptions)
        if not descs or all(d is None for d in descs):
            raise RuntimeError("This GeoTIFF has no band descriptions; cannot parse platform/date/pol.")

        parsed = []
        for band_i, desc in enumerate(descs, start=1):
            info = parse_band(desc)
            if info is None:
                continue
            info["band"] = band_i
            parsed.append(info)

        parsed_plat = [
            b for b in parsed
            if b["platform"] == PLATFORM and b["pol"] in ("VV", "VH") and b["dt"] is not None
        ]
        if not parsed_plat:
            raise RuntimeError(f"No {PLATFORM} VV/VH bands found for this GeoTIFF.")

        dts_all = find_dates_available(parsed_plat)

        dts_window = sorted([dt for dt in dts_all if start_date <= dt.date() <= end_date])
        dts_ref = sorted([dt for dt in dts_all if ref_start <= dt.date() <= ref_end])

        print(f"\nHIST-DIST: window dates = {len(dts_window)}  | baseline pre-flood dates = {len(dts_ref)}")

        for k0 in k_list:
            print(f"\nHIST-DIST: Building baseline histograms for k={k0} ...")

            vv_refs = []
            vh_refs = []

            for dt in dts_ref:
                VH, VV = get_vh_vv_for_datetime(src, parsed_plat, dt)
                if k0 > 1:
                    VH = box_mean(VH, k0)
                    VV = box_mean(VV, k0)

                vv_refs.append(hist_prob(VV, vv_bins))
                vh_refs.append(hist_prob(VH, vh_bins))

            if not vv_refs or not vh_refs:
                print(f"HIST-DIST: Skipping k={k0} (no baseline histograms).")
                continue

            VV_base = np.mean(np.stack(vv_refs, axis=0), axis=0)
            VH_base = np.mean(np.stack(vh_refs, axis=0), axis=0)

            print(f"HIST-DIST: Scoring all dates for k={k0} ...")
            for dt in dts_window:
                VH, VV = get_vh_vv_for_datetime(src, parsed_plat, dt)
                if k0 > 1:
                    VH = box_mean(VH, k0)
                    VV = box_mean(VV, k0)

                VV_t = hist_prob(VV, vv_bins)
                VH_t = hist_prob(VH, vh_bins)

                if use_js:
                    vv_score = js_divergence(VV_t, VV_base)
                    vh_score = js_divergence(VH_t, VH_base)
                else:
                    vv_score = l1_distance(VV_t, VV_base)
                    vh_score = l1_distance(VH_t, VH_base)

                rows.append({"date": dt.date(), "k": k0, "pol": "VV", "score": vv_score})
                rows.append({"date": dt.date(), "k": k0, "pol": "VH", "score": vh_score})
                rows.append({"date": dt.date(), "k": k0, "pol": "VV+VH", "score": (vv_score + vh_score)/2.0})

    return pd.DataFrame(rows)

def plot_hist_score(df: pd.DataFrame, pol: str = "VV"):
    sub = df[df["pol"] == pol].copy()
    if sub.empty:
        print(f"No score rows for pol={pol}")
        return

    for k0 in sorted(sub["k"].unique()):
        s2 = sub[sub["k"] == k0].sort_values("date")
        plt.figure(figsize=(12, 4))
        plt.plot(s2["date"], s2["score"])
        plt.title(f"{pol}: Histogram distance to pre-flood baseline (k={k0})")
        plt.xlabel("Date")
        plt.ylabel("Change score (higher = more flood-like)")
        plt.tight_layout()
        plt.show()


# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Flood Detection Pipeline - Sentinel-1 SAR (Numeric Change Summaries)")
    print("=" * 70)
    print(f"Input file: {tif_path}")
    print(f"Platform: {PLATFORM}")
    print(f"Before target date: {before_target_date}")
    print(f"During target date: {during_target_date}")
    print(f"Box size: {k}×{k}")
    print("=" * 70)
    print()

    M_raw_before, M_raw_during, M_box_before, M_box_during, acquisition_dates, before_dt, during_dt = \
        process_flood_data(tif_path)

    # --- Extract channels (raw) ---
    VH_before = M_raw_before[:, :, 0]
    VV_before = M_raw_before[:, :, 1]
    VH_during = M_raw_during[:, :, 0]
    VV_during = M_raw_during[:, :, 1]

    dVH = VH_during - VH_before
    dVV = VV_during - VV_before

    dt_label = f"{before_dt.date()}_to_{during_dt.date()}"

    # --- Numeric summaries (RAW) ---
    rows = []
    rows.append(stats_row(VV_before, "VV_before", "raw", str(before_dt.date())))
    rows.append(stats_row(VV_during, "VV_during", "raw", str(during_dt.date())))
    rows.append(stats_row(dVV,       "dVV",       "raw_change", dt_label))

    rows.append(stats_row(VH_before, "VH_before", "raw", str(before_dt.date())))
    rows.append(stats_row(VH_during, "VH_during", "raw", str(during_dt.date())))
    rows.append(stats_row(dVH,       "dVH",       "raw_change", dt_label))

    # "Flooded vs not flooded" WITHOUT thresholds:
    rows.extend(quantile_slice_stats(dVV, TAIL_PCT, "dVV", dt_label))
    rows.extend(quantile_slice_stats(dVH, TAIL_PCT, "dVH", dt_label))

    df_raw = pd.DataFrame(rows)
    print_df(df_raw, "NUMERIC SUMMARY (RAW): before, during, change + tail-split contrast")

    # --- Numeric summaries (BOX-AVERAGED) ---
    VH_before_b = M_box_before[:, :, 0]
    VV_before_b = M_box_before[:, :, 1]
    VH_during_b = M_box_during[:, :, 0]
    VV_during_b = M_box_during[:, :, 1]

    dVH_b = VH_during_b - VH_before_b
    dVV_b = VV_during_b - VV_before_b

    rows_b = []
    rows_b.append(stats_row(VV_before_b, "VV_before", "box", str(before_dt.date())))
    rows_b.append(stats_row(VV_during_b, "VV_during", "box", str(during_dt.date())))
    rows_b.append(stats_row(dVV_b,       "dVV",       "box_change", dt_label))

    rows_b.append(stats_row(VH_before_b, "VH_before", "box", str(before_dt.date())))
    rows_b.append(stats_row(VH_during_b, "VH_during", "box", str(during_dt.date())))
    rows_b.append(stats_row(dVH_b,       "dVH",       "box_change", dt_label))

    rows_b.extend(quantile_slice_stats(dVV_b, TAIL_PCT, "dVV", dt_label))
    rows_b.extend(quantile_slice_stats(dVH_b, TAIL_PCT, "dVH", dt_label))

    df_box = pd.DataFrame(rows_b)
    print_df(df_box, "NUMERIC SUMMARY (BOX): before, during, change + tail-split contrast")

    # --- Export table(s) ---
    if EXPORT_SUMMARY_CSV:
        df_out = pd.concat([df_raw.assign(which="raw"), df_box.assign(which="box")], ignore_index=True)
        df_out.to_csv(summary_csv_path, index=False)
        print(f"\nSaved CSV summary to: {summary_csv_path}")

    # Optional plots (raw + box images)
    if SHOW_PLOTS:
        print("\nGenerating visualizations...")
        visualize_flood_comparison(M_raw_before, M_raw_during, M_box_before, M_box_during, before_dt, during_dt)

    print("\nDone main pipeline.")

    # ============================================================================
    # OPTIONAL EXECUTION: Fixed dB bin time series
    # ============================================================================
    if RUN_BIN_TIMESERIES:
        print("\n" + "=" * 70)
        print("RUNNING ADD-ON: Fixed dB bin time series (percent of pixels per bin)")
        print("=" * 70)
        print(f"Window: {ts_start_date} .. {ts_end_date}")
        print(f"k_list: {k_list}")
        print(f"VH bins: {VH_BINS}")
        print(f"VV bins: {VV_BINS}")

        df_bins = build_binned_timeseries(
            tif_path=tif_path,
            k_list=k_list,
            start_date=ts_start_date,
            end_date=ts_end_date,
            vh_bins=VH_BINS,
            vv_bins=VV_BINS
        )

        print("\nBIN-TS: Head of table:")
        with pd.option_context("display.width", 180):
            print(df_bins.head(12))

        if EXPORT_BIN_TS_CSV:
            df_bins.to_csv(bin_ts_csv_path, index=False)
            print(f"\nSaved bin time series CSV to: {bin_ts_csv_path}")

        if SHOW_PLOTS:
            # Existing full-bin plots (VV then VH)
            plot_bins_over_time(df_bins, pol="VV")
            plot_bins_over_time(df_bins, pol="VH")

            # ------------------------------------------------------------
            # NEW: VV-only “isolated” plots you asked for
            #
            # NOTE on colors in your existing plot order:
            # bins are sorted by lower edge, then plotted in that order:
            #   1st=blue  -> "[-25,-21)"
            #   2nd=orange-> "[-21,-17)"
            #   4th=red   -> "[-13,-9)"
            # ------------------------------------------------------------
            RED_BIN    = "[-13,-9)"
            ORANGE_BIN = "[-21,-17)"
            BLUE_BIN   = "[-25,-21)"

            # Isolate the “red” bin line (per k)
            plot_vv_single_bin(df_bins, RED_BIN)

            # Isolate the “orange + blue” lines (and also plot their sum)
            plot_vv_two_bins(df_bins, BLUE_BIN, ORANGE_BIN, also_plot_sum=True)

        print("\nADD-ON bin time series done.")

    # ============================================================================
    # OPTIONAL EXECUTION: Histogram-distance flood score curves
    # ============================================================================
    if RUN_HIST_DISTANCE:
        print("\n" + "=" * 70)
        print("RUNNING ADD-ON METHOD 2: Histogram Distance (one flood score curve)")
        print("=" * 70)

        df_score = build_hist_distance_timeseries(
            tif_path=tif_path,
            k_list=k_list,
            start_date=ts_start_date,
            end_date=ts_end_date,
            flood_date=flood_center_date,
            vv_bins=VV_BINS,
            vh_bins=VH_BINS,
            use_js=True  # True=Jensen–Shannon, False=L1
        )

        print("\nHIST-DIST: Head of score table:")
        with pd.option_context("display.width", 180):
            print(df_score.head(12))

        if EXPORT_HIST_SCORE_CSV:
            df_score.to_csv(hist_score_csv_path, index=False)
            print(f"\nSaved histogram-distance scores CSV to: {hist_score_csv_path}")

        if SHOW_PLOTS:
            plot_hist_score(df_score, pol="VV")
            plot_hist_score(df_score, pol="VH")
            plot_hist_score(df_score, pol="VV+VH")

        print("\nHIST-DIST add-on done.")
