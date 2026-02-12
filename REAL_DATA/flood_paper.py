#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Near-real-time flood detection with Bayesian Online Changepoint Detection (BOCPD)
on Sentinel-1 SAR VV/VH time series stored as a multi-band GeoTIFF.

Implements the workflow described in:
"Potential Application of Bayesian Changepoint Detection for Near-Real-Time Flood Monitoring
 Using Sentinel-1 SAR Data" (IGARSS 2024).  (See provided PDF in this project.)

Key output:
- Flood probability map = P(r_T = 0 | x_1:T) per pixel, i.e., posterior probability that the
  *latest* observation starts a new segment (changepoint at the newest time step).

Dependencies (only):
- numpy, rasterio, scipy, matplotlib (matplotlib not required unless you add plots)

Author: (you)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import List, Literal, Optional, Sequence, Tuple

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.special import gammaln, logsumexp


Mode = Literal["VV", "VH", "VVVH_SUM"]


# =============================================================================
# Configuration dataclasses
# =============================================================================

@dataclass(frozen=True)
class TimeWindow:
    """Near-real-time window: ~1 year before flood up to closest acquisition during flood."""
    flood_date: date
    days_before: int = 365  # "approximately 1 year BEFORE" as in the paper


@dataclass(frozen=True)
class Hazard:
    """
    Constant hazard for BOCPD: H = 1/k.

    Interpretation:
    - k is the expected segment length (in number of acquisitions).
    - smaller k -> higher hazard -> more sensitive / more frequent changepoints
    - larger k -> lower hazard -> more conservative
    """
    k: float

    @property
    def H(self) -> float:
        if self.k <= 0:
            raise ValueError("Hazard parameter k must be > 0.")
        return 1.0 / self.k


@dataclass(frozen=True)
class NIGPrior:
    """
    Normal-Inverse-Gamma prior for unknown Gaussian mean and variance.

    Model:
      x ~ Normal(mu, sigma^2)
      mu | sigma^2 ~ Normal(mu0, sigma^2 / kappa0)
      sigma^2 ~ Inv-Gamma(alpha0, beta0)

    Conjugate updates yield a Student-t predictive distribution.
    """
    mu0: float = 0.0
    kappa0: float = 1.0
    alpha0: float = 2.0
    beta0: float = 1.0


@dataclass(frozen=True)
class IOConfig:
    in_tif: str
    out_prob_tif: str
    out_mask_tif: str
    chunk_rows: int = 32
    top_percentile: float = 90.0  # binary mask keeps top X percentile of flood probabilities


@dataclass(frozen=True)
class BOCPDConfig:
    mode: Mode = "VVVH_SUM"
    hazard: Hazard = Hazard(k=5.0)
    prior: NIGPrior = NIGPrior(mu0=0.0, kappa0=1.0, alpha0=2.0, beta0=1.0)
    max_run_length: Optional[int] = None  # if None, uses T (exact)


# =============================================================================
# Utilities: date parsing and band layout detection
# =============================================================================

_DATE_PATTERNS = [
    # 2019-09-17 or 2019_09_17
    re.compile(r"(?P<y>\d{4})[-_](?P<m>\d{2})[-_](?P<d>\d{2})"),
    # 20190917
    re.compile(r"(?P<y>\d{4})(?P<m>\d{2})(?P<d>\d{2})"),
]


def _parse_date_from_text(text: str) -> Optional[date]:
    if not text:
        return None
    for pat in _DATE_PATTERNS:
        m = pat.search(text)
        if m:
            y, mm, dd = int(m.group("y")), int(m.group("m")), int(m.group("d"))
            try:
                return date(y, mm, dd)
            except ValueError:
                return None
    return None


def infer_dates_from_band_descriptions(
    band_descriptions: Sequence[Optional[str]],
) -> List[Optional[date]]:
    """
    Attempt to parse acquisition dates from each band's description.
    Returns list of length = band count, possibly containing None.
    """
    out: List[Optional[date]] = []
    for desc in band_descriptions:
        out.append(_parse_date_from_text(desc or ""))
    return out


def detect_layout_and_indices(
    band_descriptions: Sequence[Optional[str]],
    total_bands: int,
) -> Tuple[str, List[int], List[int]]:
    """
    Detect layout:
      - "interleaved": [VV_t1, VH_t1, VV_t2, VH_t2, ...]
      - "blocked":     [VV_t1..VV_tT, VH_t1..VH_tT]

    Uses band descriptions if possible:
      - If band1 mentions VV and band2 mentions VH -> interleaved
      - If first half mostly VV and second half mostly VH -> blocked
    Else defaults to interleaved.

    Returns:
      (layout_name, vv_band_indices_1based, vh_band_indices_1based)
    """
    if total_bands % 2 != 0:
        raise ValueError(f"Expected an even number of bands (VV/VH pairs), got {total_bands}.")

    def tok(desc: Optional[str]) -> str:
        return (desc or "").upper()

    if total_bands >= 2:
        d1, d2 = tok(band_descriptions[0]), tok(band_descriptions[1])
        if ("VV" in d1) and ("VH" in d2):
            vv = list(range(1, total_bands + 1, 2))
            vh = list(range(2, total_bands + 1, 2))
            return "interleaved", vv, vh

    half = total_bands // 2
    first = [tok(band_descriptions[i]) for i in range(0, half)]
    second = [tok(band_descriptions[i]) for i in range(half, total_bands)]

    first_vv = sum("VV" in s for s in first)
    first_vh = sum("VH" in s for s in first)
    second_vv = sum("VV" in s for s in second)
    second_vh = sum("VH" in s for s in second)

    if first_vv >= max(1, half // 2) and second_vh >= max(1, half // 2) and first_vh == 0 and second_vv == 0:
        vv = list(range(1, half + 1))
        vh = list(range(half + 1, total_bands + 1))
        return "blocked", vv, vh

    vv = list(range(1, total_bands + 1, 2))
    vh = list(range(2, total_bands + 1, 2))
    return "interleaved", vv, vh


def build_time_steps(
    vv_band_indices_1based: Sequence[int],
    vh_band_indices_1based: Sequence[int],
    band_dates: Sequence[Optional[date]],
    user_dates: Optional[Sequence[date]],
) -> List[date]:
    """
    Build acquisition date list per time step t=1..T.

    Priority:
    1) If dates can be inferred from VV (or VH) band descriptions for each time step, use them.
    2) Else require user_dates.

    Validates length == T.
    """
    T = len(vv_band_indices_1based)
    if len(vh_band_indices_1based) != T:
        raise ValueError("VV and VH index lists must have the same length (T).")

    inferred: List[Optional[date]] = []
    for bi in vv_band_indices_1based:
        inferred.append(band_dates[bi - 1])

    if all(d is not None for d in inferred):
        return [d for d in inferred if d is not None]  # type: ignore[return-value]

    inferred2: List[Optional[date]] = []
    for bi in vh_band_indices_1based:
        inferred2.append(band_dates[bi - 1])
    if all(d is not None for d in inferred2):
        return [d for d in inferred2 if d is not None]  # type: ignore[return-value]

    if user_dates is None:
        raise ValueError(
            "Could not infer acquisition dates from band descriptions/tags. "
            "Please provide user_dates (list of datetime.date) with length T."
        )
    if len(user_dates) != T:
        raise ValueError(f"user_dates length ({len(user_dates)}) must equal T ({T}).")
    return list(user_dates)


def select_near_real_time_window(
    dates: Sequence[date],
    window: TimeWindow,
) -> np.ndarray:
    """
    Paper-like near-real-time window:
      - start = flood_date - ~1 year
      - end = closest acquisition date <= flood_date
      - keep acquisitions within [start, end]
    """
    if len(dates) == 0:
        raise ValueError("No acquisition dates provided.")

    start_date = window.flood_date - timedelta(days=window.days_before)
    eligible = [d for d in dates if d <= window.flood_date]
    if not eligible:
        raise ValueError("No acquisitions on or before flood_date; cannot form near-real-time window.")
    end_date = max(eligible)

    mask = np.array([(d >= start_date) and (d <= end_date) for d in dates], dtype=bool)
    if mask.sum() < 2:
        raise ValueError(
            f"Window selection produced too few time steps ({mask.sum()}). "
            f"Check dates and flood_date. start={start_date}, end={end_date}"
        )
    return mask


# =============================================================================
# BOCPD math (vectorized across pixels) with Normal-Inverse-Gamma prior
# =============================================================================

def student_t_logpdf_from_nig(
    x: np.ndarray,
    mu: np.ndarray,
    kappa: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    nu = 2.0 * alpha
    scale2 = beta * (kappa + 1.0) / (alpha * kappa)
    scale2 = np.maximum(scale2, 1e-12)
    z2 = (x - mu) ** 2 / scale2

    half = 0.5
    log_norm = (
        gammaln((nu + 1.0) * half)
        - gammaln(nu * half)
        - half * (np.log(nu) + np.log(np.pi))
        - half * np.log(scale2)
    )
    log_kernel = -((nu + 1.0) * half) * np.log1p(z2 / nu)
    return log_norm + log_kernel


def nig_posterior_update(
    x: np.ndarray,
    mu: np.ndarray,
    kappa: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    kappa_new = kappa + 1.0
    mu_new = (kappa * mu + x) / kappa_new
    alpha_new = alpha + 0.5
    beta_new = beta + 0.5 * (kappa * (x - mu) ** 2) / kappa_new
    return mu_new, kappa_new, alpha_new, beta_new


def bocpd_latest_changepoint_prob(
    x: np.ndarray,
    hazard: Hazard,
    prior: NIGPrior,
    max_run_length: Optional[int] = None,
) -> np.ndarray:
    """
    Compute P(r_T = 0 | x_1:T) per pixel for a chunk.

    x: (T, N) float, may contain NaN.
    """
    T, N = x.shape
    Rmax = T if (max_run_length is None) else int(max_run_length)
    Rmax = min(Rmax, T)

    H = hazard.H
    logH = np.log(H)
    log1mH = np.log(1.0 - H)

    logR = -np.inf * np.ones((Rmax + 1, N), dtype=np.float64)
    logR[0, :] = 0.0

    mu = np.full((Rmax + 1, N), prior.mu0, dtype=np.float64)
    kappa = np.full((Rmax + 1, N), prior.kappa0, dtype=np.float64)
    alpha = np.full((Rmax + 1, N), prior.alpha0, dtype=np.float64)
    beta = np.full((Rmax + 1, N), prior.beta0, dtype=np.float64)

    for t in range(T):
        xt = x[t, :]
        valid = np.isfinite(xt)

        log_pred = np.zeros((Rmax + 1, N), dtype=np.float64)
        if np.any(valid):
            log_pred[:, valid] = student_t_logpdf_from_nig(
                xt[valid][None, :],
                mu[:, valid],
                kappa[:, valid],
                alpha[:, valid],
                beta[:, valid],
            )

        new_logR = -np.inf * np.ones_like(logR)
        new_logR[1:, :] = logR[:-1, :] + log1mH + log_pred[:-1, :]
        new_logR[0, :] = logsumexp(logR + logH + log_pred, axis=0)

        norm = logsumexp(new_logR, axis=0)
        new_logR = new_logR - norm[None, :]

        new_mu = mu.copy()
        new_kappa = kappa.copy()
        new_alpha = alpha.copy()
        new_beta = beta.copy()

        if np.any(valid):
            mu0 = np.full((N,), prior.mu0, dtype=np.float64)
            k0 = np.full((N,), prior.kappa0, dtype=np.float64)
            a0 = np.full((N,), prior.alpha0, dtype=np.float64)
            b0 = np.full((N,), prior.beta0, dtype=np.float64)

            mu_up, k_up, a_up, b_up = nig_posterior_update(
                xt[valid], mu0[valid], k0[valid], a0[valid], b0[valid]
            )
            new_mu[0, valid] = mu_up
            new_kappa[0, valid] = k_up
            new_alpha[0, valid] = a_up
            new_beta[0, valid] = b_up

            mu_g, k_g, a_g, b_g = nig_posterior_update(
                xt[valid][None, :],
                mu[:-1, valid],
                kappa[:-1, valid],
                alpha[:-1, valid],
                beta[:-1, valid],
            )
            new_mu[1:, valid] = mu_g
            new_kappa[1:, valid] = k_g
            new_alpha[1:, valid] = a_g
            new_beta[1:, valid] = b_g

        logR = new_logR
        mu, kappa, alpha, beta = new_mu, new_kappa, new_alpha, new_beta

    return np.exp(logR[0, :]).astype(np.float32)


# =============================================================================
# Raster IO: read chunked VV/VH time series and write outputs
# =============================================================================

def _read_time_series_chunk(
    src: rasterio.io.DatasetReader,
    band_indices_1based: Sequence[int],
    window: Window,
) -> np.ndarray:
    """
    Read selected bands for a window.
    Returns (T, rows, cols) float32 with nodata -> NaN.
    """
    T = len(band_indices_1based)
    rows = int(window.height)
    cols = int(window.width)
    out = np.empty((T, rows, cols), dtype=np.float32)

    nodata = src.nodata
    for i, b in enumerate(band_indices_1based):
        arr = src.read(b, window=window).astype(np.float32, copy=False)
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)
        out[i, :, :] = arr

    return out


def _streaming_histogram_update(hist: np.ndarray, values: np.ndarray) -> None:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return
    v = np.clip(v, 0.0, 1.0)
    nbins = hist.size
    idx = np.floor(v * (nbins - 1)).astype(np.int64)
    np.add.at(hist, idx, 1)


def _histogram_percentile_threshold(hist: np.ndarray, top_percentile: float) -> float:
    if not (0.0 < top_percentile < 100.0):
        raise ValueError("top_percentile must be in (0, 100).")

    total = hist.sum()
    if total == 0:
        return 1.0

    keep = top_percentile / 100.0
    target_keep = int(np.ceil(total * keep))

    cumsum_from_top = np.cumsum(hist[::-1])
    j = int(np.searchsorted(cumsum_from_top, target_keep, side="left"))
    nbins = hist.size

    bin_from_bottom = (nbins - 1) - j
    thr = bin_from_bottom / (nbins - 1)
    return float(thr)


def run_pipeline(
    io_cfg: IOConfig,
    bocpd_cfg: BOCPDConfig,
    window_cfg: TimeWindow,
    user_dates: Optional[Sequence[date]] = None,
) -> None:
    with rasterio.open(io_cfg.in_tif) as src:
        total_bands = src.count
        band_desc = list(src.descriptions) if src.descriptions is not None else [None] * total_bands

        layout, vv_idx, vh_idx = detect_layout_and_indices(band_desc, total_bands)
        band_dates = infer_dates_from_band_descriptions(band_desc)
        dates = build_time_steps(vv_idx, vh_idx, band_dates, user_dates)

        use_mask = select_near_real_time_window(dates, window_cfg)
        vv_sel = [b for b, m in zip(vv_idx, use_mask) if m]
        vh_sel = [b for b, m in zip(vh_idx, use_mask) if m]
        dates_sel = [d for d, m in zip(dates, use_mask) if m]
        T = len(dates_sel)

        if T < 2:
            raise ValueError("Selected time window has <2 acquisitions; cannot run changepoint detection.")

        profile = src.profile.copy()
        profile.update(
            count=1,
            dtype="float32",
            nodata=None,
            compress="deflate",
        )
        

        hist_bins = 10000
        hist = np.zeros((hist_bins,), dtype=np.int64)

        with rasterio.open(io_cfg.out_prob_tif, "w", **profile) as dst_prob:
            H_img, W_img = src.height, src.width

            for row0 in range(0, H_img, io_cfg.chunk_rows):
                nrows = min(io_cfg.chunk_rows, H_img - row0)
                win = Window(col_off=0, row_off=row0, width=W_img, height=nrows)

                vv_cube = _read_time_series_chunk(src, vv_sel, win)  # (T, nrows, W)
                vh_cube = _read_time_series_chunk(src, vh_sel, win)

                if bocpd_cfg.mode == "VV":
                    x_cube = vv_cube
                elif bocpd_cfg.mode == "VH":
                    x_cube = vh_cube
                elif bocpd_cfg.mode == "VVVH_SUM":
                    x_cube = vv_cube + vh_cube
                else:
                    raise ValueError(f"Unknown mode: {bocpd_cfg.mode}")

                N = nrows * W_img
                x = x_cube.reshape(T, N)

                p0 = bocpd_latest_changepoint_prob(
                    x=x,
                    hazard=bocpd_cfg.hazard,
                    prior=bocpd_cfg.prior,
                    max_run_length=bocpd_cfg.max_run_length,
                )

                p0_img = p0.reshape(nrows, W_img)
                dst_prob.write(p0_img.astype(np.float32), 1, window=win)

                _streaming_histogram_update(hist, p0)

        thr = _histogram_percentile_threshold(hist, io_cfg.top_percentile)

        
        mask_profile = profile.copy()
        mask_profile.update(dtype="uint8", nodata=0, compress="deflate")
        mask_profile.pop("predictor", None)  
        with rasterio.open(io_cfg.out_prob_tif) as src_prob, rasterio.open(
            io_cfg.out_mask_tif, "w", **mask_profile
        ) as dst_mask:
            H_img, W_img = src_prob.height, src_prob.width
            for row0 in range(0, H_img, io_cfg.chunk_rows):
                nrows = min(io_cfg.chunk_rows, H_img - row0)
                win = Window(col_off=0, row_off=row0, width=W_img, height=nrows)
                prob = src_prob.read(1, window=win).astype(np.float32, copy=False)
                valid = np.isfinite(prob)
                mask = np.zeros_like(prob, dtype=np.uint8)
                mask[valid & (prob >= thr)] = 1
                dst_mask.write(mask, 1, window=win)

    print("Done.")
    print(f"Input:               {io_cfg.in_tif}")
    print(f"Layout detected:     {layout}")
    print(f"Time steps used:     {T}")
    print(f"Flood date:          {window_cfg.flood_date.isoformat()}")
    print(f"Output prob GeoTIFF: {io_cfg.out_prob_tif}")
    print(f"Output mask GeoTIFF: {io_cfg.out_mask_tif}")
    print(f"Top-percentile:      {io_cfg.top_percentile}%  -> threshold ~= {thr:.4f}")
    print(f"Hazard:              H=1/k, k={bocpd_cfg.hazard.k} -> H={bocpd_cfg.hazard.H:.4f}")


# =============================================================================
# main()
# =============================================================================

def main() -> None:
    """
    Example settings matching your described case.

    You said your acquisitions span: 22/09/2018 to 05/09/2020,
    and flood is on 17/09/2019. The paper uses ~1 year BEFORE the flood
    up to the closest acquisition <= the flood date (no future data).
    """
    in_tif = "/Users/lunarolland/Desktop/DATASETS/Spain_7370579_S1_VV_VH_.tif"
    flood_date = date(2019, 9, 17)

    # If band descriptions do NOT contain dates, provide them here (length T)
    user_dates: Optional[List[date]] = None

    io_cfg = IOConfig(
        in_tif=in_tif,
        out_prob_tif="bocpd_flood_probability.tif",
        out_mask_tif="bocpd_flood_mask.tif",
        chunk_rows=32,
        top_percentile=90.0,   
    )

    bocpd_cfg = BOCPDConfig(
        mode="VVVH_SUM",
        hazard=Hazard(k=5.0),
        prior=NIGPrior(mu0=0.0, kappa0=1.0, alpha0=2.0, beta0=1.0),
        max_run_length=None,
    )

    window_cfg = TimeWindow(flood_date=flood_date, days_before=365)

    run_pipeline(
        io_cfg=io_cfg,
        bocpd_cfg=bocpd_cfg,
        window_cfg=window_cfg,
        user_dates=user_dates,
    )


if __name__ == "__main__":
    main()
