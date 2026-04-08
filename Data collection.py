"""
CE 295 - Data Science for Energy
Step 1: Data Collection & EDA
Source: NREL NSRDB — Local CSV files — Los Angeles, CA
Authors: Antoine & Matthieu

INSTRUCTIONS:
    1. Update CSV_DIR below with the path to your folder
    2. Files must be named: 204191_34.05_-118.26_2021.csv (change year)
    3. Run: python data_collection.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os

# ─────────────────────────────────────────
# CONFIG — update CSV_DIR with your path
# ─────────────────────────────────────────
CSV_DIR    = r"C:\Users\antoi\Documents\Berkeley\Courses\Data Science in Energy\Project\Données"
OUTPUT_DIR = os.path.join(CSV_DIR, "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FILE_PREFIX = "204191_34.05_-118.26_"
YEARS       = [2021, 2022, 2023, 2024]
TRAIN_YEARS = [2021, 2022]
VAL_YEAR    = 2023
TEST_YEAR   = 2024


# ─────────────────────────────────────────
# 1. LOAD LOCAL CSV FILES
# ─────────────────────────────────────────
def load_year(year: int) -> pd.DataFrame:
    path = os.path.join(CSV_DIR, f"{FILE_PREFIX}{year}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\nFile not found: {path}"
            f"\nMake sure the CSV is in: {os.path.abspath(CSV_DIR)}"
        )
    # NSRDB CSVs have 2 metadata rows before column headers
    df = pd.read_csv(path, skiprows=2)
    df["datetime"] = pd.to_datetime({
        "year":   df["Year"],
        "month":  df["Month"],
        "day":    df["Day"],
        "hour":   df["Hour"],
        "minute": df["Minute"],
    })
    df = df.set_index("datetime")
    df = df.drop(columns=["Year", "Month", "Day", "Hour", "Minute"], errors="ignore")
    print(f"  Loaded {year}: {len(df):,} rows")
    return df


def load_all_years() -> pd.DataFrame:
    frames = [load_year(y) for y in YEARS]
    combined = pd.concat(frames).sort_index()
    combined.to_csv(f"{OUTPUT_DIR}/nsrdb_all_raw.csv")
    print(f"  Combined: {len(combined):,} rows → {OUTPUT_DIR}/nsrdb_all_raw.csv")
    return combined


# ─────────────────────────────────────────
# 2. PREPROCESSING & FEATURE ENGINEERING
# ─────────────────────────────────────────
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Rename columns to lowercase standard names
    rename_map = {
        "GHI":               "ghi",
        "DNI":               "dni",
        "DHI":               "dhi",
        "Temperature":       "temp",
        "Wind Speed":        "wind_speed",
        "Relative Humidity": "humidity",
        "Cloud Type":        "cloud_type",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Target: PV power proxy P = eta * GHI (eta = 0.18)
    df["solar_power"] = (df["ghi"] * 0.18).clip(lower=0)

    # Clip negative irradiance (sensor noise at night)
    for col in ["ghi", "dni", "dhi"]:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    # Cyclical time features
    df["hour_sin"]  = np.sin(2 * np.pi * df.index.hour / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df.index.hour / 24)
    df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)
    df["dayofyear"] = df.index.dayofyear / 365.0
    df["is_daytime"] = (df["ghi"] > 0).astype(int)

    df = df.ffill().dropna()
    return df


def normalise(df: pd.DataFrame) -> tuple:
    cols = [c for c in ["ghi", "dni", "dhi", "temp", "wind_speed", "humidity", "solar_power"]
            if c in df.columns]
    scaler = {}
    for col in cols:
        mn, mx = df[col].min(), df[col].max()
        df[col] = (df[col] - mn) / (mx - mn + 1e-8)
        scaler[col] = {"min": float(mn), "max": float(mx)}
    with open(f"{OUTPUT_DIR}/scaler.json", "w") as f:
        json.dump(scaler, f, indent=2)
    print(f"  Scaler saved → {OUTPUT_DIR}/scaler.json")
    return df, scaler


# ─────────────────────────────────────────
# 3. SUMMARY & EDA PLOTS
# ─────────────────────────────────────────
def print_summary(df: pd.DataFrame):
    print("\n── Dataset Summary ───────────────────────────────")
    print(f"  Period  : {df.index[0]}  →  {df.index[-1]}")
    print(f"  Rows    : {len(df):,}")
    print(f"  Columns : {list(df.columns)}")
    print(f"  Missing : {df.isnull().sum().sum()}")
    print(f"  Avg GHI : {df['ghi'].mean():.4f} (normalised)")
    print(f"  Max GHI : {df['ghi'].max():.4f} (normalised)")
    print("──────────────────────────────────────────────────\n")


def plot_eda(df: pd.DataFrame):
    """3 EDA plots on the preprocessed (but not normalised) dataframe."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle("NREL NSRDB — Los Angeles, CA (2021–2024) | EDA", fontsize=13, fontweight="bold")

    # Plot 1: GHI over 2 sample weeks
    sample = df["2022-06-01":"2022-06-14"]
    axes[0].plot(sample.index, sample["ghi"], color="#F4A261", linewidth=1)
    axes[0].set_title("Global Horizontal Irradiance (GHI) — June 2022")
    axes[0].set_ylabel("GHI (W/m²)")
    axes[0].grid(alpha=0.3)

    # Plot 2: Monthly average GHI
    monthly = df["ghi"].resample("ME").mean()
    axes[1].bar(monthly.index, monthly.values, color="#2A9D8F", width=20)
    axes[1].set_title("Monthly Average GHI (2021–2024)")
    axes[1].set_ylabel("Avg GHI (W/m²)")
    axes[1].grid(alpha=0.3, axis="y")

    # Plot 3: Average diurnal profile
    diurnal = df.groupby(df.index.hour)["ghi"].mean()
    axes[2].plot(diurnal.index, diurnal.values, color="#264653", linewidth=2, marker="o", markersize=4)
    axes[2].fill_between(diurnal.index, diurnal.values, alpha=0.2, color="#264653")
    axes[2].set_title("Average Diurnal GHI Profile")
    axes[2].set_xlabel("Hour of day")
    axes[2].set_ylabel("Avg GHI (W/m²)")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/eda_plots.png", dpi=150)
    print(f"  EDA plots saved → {OUTPUT_DIR}/eda_plots.png")
    plt.show()


# ─────────────────────────────────────────
# 4. TRAIN / VAL / TEST SPLIT
# ─────────────────────────────────────────
def split_dataset(df: pd.DataFrame) -> tuple:
    """
    Chronological split — never shuffle time series!
      Train : 2021–2022
      Val   : 2023
      Test  : 2024
    """
    train = df[df.index.year.isin(TRAIN_YEARS)]
    val   = df[df.index.year == VAL_YEAR]
    test  = df[df.index.year == TEST_YEAR]
    print(f"  Train : {len(train):,} rows  ({train.index[0].date()} → {train.index[-1].date()})")
    print(f"  Val   : {len(val):,} rows  ({val.index[0].date()} → {val.index[-1].date()})")
    print(f"  Test  : {len(test):,} rows  ({test.index[0].date()} → {test.index[-1].date()})")
    return train, val, test


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("\n=== CE 295 — Step 1: Data Collection & EDA ===\n")

    # 1. Load raw CSV files
    print("[1/4] Loading NSRDB CSV files...")
    df_raw = load_all_years()

    # 2. Preprocess (rename, clip, feature engineering)
    print("[2/4] Preprocessing & feature engineering...")
    df_clean = preprocess(df_raw.copy())

    # 3. EDA on clean (pre-normalisation) data — GHI still in W/m²
    print("[3/4] Generating EDA plots...")
    plot_eda(df_clean)

    # 4. Normalise
    print("[4/4] Normalising and splitting...")
    df_norm, scaler = normalise(df_clean.copy())
    print_summary(df_norm)
    df_norm.to_csv(f"{OUTPUT_DIR}/nsrdb_processed.csv")
    print(f"  Processed dataset saved → {OUTPUT_DIR}/nsrdb_processed.csv")

    # 5. Split
    train, val, test = split_dataset(df_norm)
    train.to_csv(f"{OUTPUT_DIR}/train.csv")
    val.to_csv(f"{OUTPUT_DIR}/val.csv")
    test.to_csv(f"{OUTPUT_DIR}/test.csv")

    print("\n✓ Data pipeline complete. Ready for model training.")
    print("  Next step → run: arx_model.py  then  lstm_model.py\n")