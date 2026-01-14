#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CUSUM with a burn-in (Phase I / Phase II) period.

Phase I (burn-in): estimate baseline mean and std from the first B observations.
Phase II (monitoring): run a two-sided CUSUM using the frozen baseline estimates.

This script also simulates ARL0 and ARL1 (conditional delay) like your earlier code,
but now using the burn-in-based detector.
"""

import numpy as np
import pandas as pd


# ----------------------------
# PHASE I: baseline estimation
# ----------------------------
def estimate_baseline(x: np.ndarray) -> tuple[float, float]:
    """Estimate baseline mean and std from burn-in data x (assumed in-control)."""
    if x.ndim != 1:
        raise ValueError("x must be a 1D array")

    B_local = len(x)
    if B_local < 2:
        raise ValueError("Burn-in sample too small")

    mu_hat = float(np.mean(x))
    sigma_hat = float(np.std(x, ddof=1))  # unbiased sample std

    if sigma_hat <= 0:
        raise ValueError("Estimated sigma must be > 0")

    return mu_hat, sigma_hat


# ----------------------------
# PHASE II: monitoring CUSUM
# ----------------------------
def cusum_alarm_time_burnin(x: np.ndarray, B: int, k: float, h: float) -> int | None:
    """
    Two-sided CUSUM with a burn-in (Phase I) period.

    Returns:
        Alarm time as a GLOBAL index t in {0, ..., len(x)-1}, or None if no alarm.

    What it does:
    - Phase I: uses x[:B] to estimate mu_hat and sigma_hat (baseline). No alarms here.
    - Phase II: runs the usual two-sided CUSUM on x[B:] using frozen estimates.
    """
    if x.ndim != 1:
        raise ValueError("x must be a 1D array")
    if not (0 < B < len(x)):
        raise ValueError("B must satisfy 0 < B < len(x)")
    if h <= 0:
        raise ValueError("h must be > 0")

    # Phase I: burn-in estimation (baseline is frozen after this)
    mu_hat, sigma_hat = estimate_baseline(x[:B])

    # Phase II: monitoring begins at time t = B
    S = 0.0  
    T = 0.0  

    for t in range(B, len(x)):
        z = (x[t] - mu_hat) / sigma_hat  

        S = max(0.0, S + z - k)          
        T = max(0.0, T - z - k)          

        if S > h or T > h:
            return t                      

    return None


# ----------------------------
# DATA generation helpers
# ----------------------------
def generate_in_control(T: int, mu0: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Generate a length-T in-control sequence."""
    return rng.normal(loc=mu0, scale=sigma, size=T)


def generate_one_change_definition_B(
    T: int,
    tau: int,
    delta: float,
    mu0: float,
    sigma: float,
    rng: np.random.Generator
) -> np.ndarray:
    """
    One change at time tau, random direction:
      X_t ~ N(mu0, sigma) for t < tau
      X_t ~ N(mu0 ± delta*sigma, sigma) for t >= tau
    """
    if not (0 < tau < T):
        raise ValueError("tau must satisfy 0 < tau < T")

    sign = rng.choice([-1.0, 1.0])
    mu1 = mu0 + sign * delta * sigma

    x = np.empty(T, dtype=float)
    x[:tau] = rng.normal(loc=mu0, scale=sigma, size=tau)
    x[tau:] = rng.normal(loc=mu1, scale=sigma, size=T - tau)
    return x


# ----------------------------
# ARL0 simulation (burn-in)
# ----------------------------
def simulate_arl0_burnin(
    n_trials: int,
    T_max: int,
    B: int,
    mu0: float,
    sigma: float,
    k: float,
    h: float,
    seed: int = 0
) -> dict:
    """
    ARL0 measured from start of monitoring (Phase II).
    Run length = number of monitored points until alarm.
    """
    rng = np.random.default_rng(seed)
    run_lengths = []
    censored = 0

    for _ in range(n_trials):
        x = generate_in_control(T=T_max, mu0=mu0, sigma=sigma, rng=rng)
        t_alarm = cusum_alarm_time_burnin(x=x, B=B, k=k, h=h)

        if t_alarm is None:
            censored += 1
            run_lengths.append(T_max - B)           # monitoring horizon length
        else:
            run_lengths.append((t_alarm - B) + 1)   # monitoring run length

    rl = np.array(run_lengths, dtype=float)
    return {
        "run_lengths": rl,
        "ARL0": float(np.mean(rl)),
        "SDRL0": float(np.std(rl, ddof=1)),
        "censored_frac": censored / n_trials
    }


# ----------------------------
# ARL1 simulation (burn-in)
# ----------------------------
def simulate_arl1_burnin(
    n_trials: int,
    T_max: int,
    B: int,
    tau: int,
    delta: float,
    mu0: float,
    sigma: float,
    k: float,
    h: float,
    seed: int = 1
) -> dict:
    """
    Conditional detection delay (ARL1-style):
      delay = t_alarm - tau, conditioned on t_alarm >= tau (detected after change).

    Also records:
      early_false_frac = P(t_alarm < tau)
      censored_frac = P(no alarm within horizon)
    """
    rng = np.random.default_rng(seed)
    delays = []
    early_false = 0
    censored = 0

    if not (0 < B < tau < T_max):
        raise ValueError("Need 0 < B < tau < T_max so the change happens after burn-in.")

    for _ in range(n_trials):
        x = generate_one_change_definition_B(
            T=T_max, tau=tau, delta=delta, mu0=mu0, sigma=sigma, rng=rng
        )

        t_alarm = cusum_alarm_time_burnin(x=x, B=B, k=k, h=h)

        if t_alarm is None:
            censored += 1
            continue

        if t_alarm < tau:
            early_false += 1
        else:
            delays.append(t_alarm - tau)

    delays = np.array(delays, dtype=float)
    return {
        "delta": delta,
        "delays": delays,
        "ARL1_delay_cond": float(np.mean(delays)) if len(delays) else np.nan,
        "SDRL1_cond": float(np.std(delays, ddof=1)) if len(delays) > 1 else np.nan,
        "early_false_frac": early_false / n_trials,
        "censored_frac": censored / n_trials,
        "n_detected_after_change": int(len(delays))
    }


# ----------------------------
# MAIN: run experiments + save table
# ----------------------------
if __name__ == "__main__":
    mu0 = 0.0
    sigma = 1.0
    B = 200
    T_max = 300_000
    tau = 400

    # Number of Monte Carlo trials
    n_trials_arl0 = 10_000
    n_trials_arl1 = 10_000

    designs = [
        {"label": "best_for_delta_0.5", "delta_star": 0.5, "k": 0.25, "h": 8.633},
        {"label": "best_for_delta_2.0", "delta_star": 2.0, "k": 1.00, "h": 2.644},
    ]

    # Shifts to evaluate under ARL1
    deltas = [0.5, 2.0]

    rows = []

    for dsg in designs:
        k = dsg["k"]
        h = dsg["h"]

        # ARL0 (in-control)
        arl0 = simulate_arl0_burnin(
            n_trials=n_trials_arl0,
            T_max=T_max,
            B=B,
            mu0=mu0,
            sigma=sigma,
            k=k,
            h=h,
            seed=999
        )

        rows.append({
            "design": dsg["label"],
            "case": "ARL0",
            "delta_star": dsg["delta_star"],
            "k": k,
            "h": h,
            "B": B,
            "tau": np.nan,
            "delta": np.nan,
            "ARL": arl0["ARL0"],
            "SDRL": arl0["SDRL0"],
            "early_false_frac": np.nan,
            "censored_frac": arl0["censored_frac"],
            "n_detected_after_change": np.nan,
        })

        # ARL1 (one change at tau)
        for delta in deltas:
            arl1 = simulate_arl1_burnin(
                n_trials=n_trials_arl1,
                T_max=T_max,
                B=B,
                tau=tau,
                delta=delta,
                mu0=mu0,
                sigma=sigma,
                k=k,
                h=h,
                seed=2020 + int(delta * 100) + int(10 * dsg["delta_star"])
            )

            rows.append({
                "design": dsg["label"],
                "case": "ARL1",
                "delta_star": dsg["delta_star"],
                "k": k,
                "h": h,
                "B": B,
                "tau": tau,
                "delta": delta,
                "ARL": arl1["ARL1_delay_cond"],
                "SDRL": arl1["SDRL1_cond"],
                "early_false_frac": arl1["early_false_frac"],
                "censored_frac": arl1["censored_frac"],
                "n_detected_after_change": arl1["n_detected_after_change"],
            })

    table = pd.DataFrame(rows).sort_values(
        by=["design", "case", "delta"],
        ascending=[True, True, True],
        na_position="last"
    )

    pd.set_option("display.width", 180)
    pd.set_option("display.max_rows", 200)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    table.to_csv("cusum_burnin_table.csv", index=False)

    latex = table.to_latex(
        index=False,
        float_format="%.3f",
        na_rep="",
        caption="CUSUM with burn-in: ARL0 (Phase II run length) and conditional ARL1 delays.",
        label="tab:cusum_burnin_table"
    )
    with open("cusum_burnin_table.tex", "w") as f:
        f.write(latex)

    print("\nSaved: cusum_burnin_table.csv and cusum_burnin_table.tex")
