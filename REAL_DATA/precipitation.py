#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 19:09:10 2026

@author: lunarolland
"""
import pandas as pd
import matplotlib.pyplot as plt

# --- 1) Load your exported CSV (update path as needed) ---
csv_path = "/Users/lunarolland/Desktop/DATASETS/Spain_7370579_daily_precip_ERA5_CHIRPS_pm1y.csv"
df = pd.read_csv(csv_path)

# --- 2) Keep only ERA5-Land ---
era5 = df[df["source"].astype(str).str.strip() == "ERA5-Land"].copy()

# --- 3) Parse and clean ---
era5["date"] = pd.to_datetime(era5["date"], errors="coerce")
era5["precip_mm"] = pd.to_numeric(era5["precip_mm"], errors="coerce")

era5 = era5.dropna(subset=["date"]).sort_values("date")
# (optional) if some precip rows are missing, keep them as gaps:
# era5 = era5.dropna(subset=["precip_mm"])

# --- 4) Plot: daily precipitation time series ---
plt.figure(figsize=(12, 4))
plt.plot(era5["date"], era5["precip_mm"])
plt.title("Daily Precipitation (ERA5-Land) — Spain_7370579")
plt.xlabel("Date")
plt.ylabel("Precipitation (mm/day)")
plt.tight_layout()
plt.show()

era5["precip_mm_7d"] = era5["precip_mm"].rolling(7, min_periods=1).mean()

plt.figure(figsize=(12, 4))
plt.plot(era5["date"], era5["precip_mm"], label="Daily")
plt.plot(era5["date"], era5["precip_mm_7d"], label="7-day mean")
plt.title("Daily Precipitation (ERA5-Land) — Spain_7370579")
plt.xlabel("Date")
plt.ylabel("Precipitation (mm/day)")
plt.legend()
plt.tight_layout()
plt.show()

