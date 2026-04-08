"""
CE 295 - Data Science for Energy
Step 2a: ARX Baseline Model for Solar Power Forecasting
Authors: Sohel & François

ARX (AutoRegressive with eXogenous inputs):
    y(t) = a1*y(t-1) + ... + na*y(t-na)
         + b1*u(t-1) + ... + nb*u(t-nb) + e(t)

    y(t) = solar_power (normalised)
    u(t) = exogenous features: temp, wind, humidity, hour_sin, hour_cos, ...
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
DATA_DIR   = r"C:\Users\antoi\Documents\Berkeley\Courses\Data Science in Energy\Project\Données\data"
OUTPUT_DIR = DATA_DIR

# ARX hyperparameters
NA = 24   # autoregressive lags of solar_power (past 24 hours)
NB = 3    # exogenous input lags
EXOG_FEATURES = [
    "temp", "wind_speed", "humidity",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "is_daytime"
]


# ─────────────────────────────────────────
# 1. BUILD ARX FEATURE MATRIX
# ─────────────────────────────────────────
def build_arx_features(df: pd.DataFrame) -> tuple:
    """
    Build regression matrix X and target y for ARX model.
    For each t: X[t] = [y(t-1)..y(t-na), u(t-1)..u(t-nb)]
    """
    max_lag = max(NA, NB)
    target  = df["solar_power"].values
    rows_X, rows_y = [], []

    for t in range(max_lag, len(df)):
        row = []
        # AR lags of target
        for lag in range(1, NA + 1):
            row.append(target[t - lag])
        # Exogenous lags
        for col in EXOG_FEATURES:
            for lag in range(1, NB + 1):
                row.append(df[col].values[t - lag])
        rows_X.append(row)
        rows_y.append(target[t])

    X     = np.array(rows_X)
    y     = np.array(rows_y)
    index = df.index[max_lag:]
    return X, y, index


# ─────────────────────────────────────────
# 2. PERSISTENCE BASELINE
# ─────────────────────────────────────────
def persistence_baseline(df: pd.DataFrame) -> dict:
    """Simplest baseline: ŷ(t) = y(t-1)."""
    y      = df["solar_power"].values
    y_pred = y[:-1]
    y_true = y[1:]
    rmse   = np.sqrt(mean_squared_error(y_true, y_pred))
    mae    = mean_absolute_error(y_true, y_pred)
    print(f"  Persistence — RMSE: {rmse:.4f} | MAE: {mae:.4f}")
    return {"rmse": rmse, "mae": mae, "y_true": y_true, "y_pred": y_pred}


# ─────────────────────────────────────────
# 3. TRAIN ARX
# ─────────────────────────────────────────
def train_arx(train: pd.DataFrame, val: pd.DataFrame) -> tuple:
    print("  Building feature matrices...")
    X_train, y_train, _   = build_arx_features(train)
    X_val,   y_val,   idx = build_arx_features(val)
    print(f"  Train matrix: {X_train.shape} | Val matrix: {X_val.shape}")

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    y_pred_val = np.clip(model.predict(X_val), 0, 1)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    mae  = mean_absolute_error(y_val, y_pred_val)
    print(f"  ARX Val  — RMSE: {rmse:.4f} | MAE: {mae:.4f}")

    return model, {"y_true": y_val, "y_pred": y_pred_val, "idx": idx, "rmse": rmse, "mae": mae}


# ─────────────────────────────────────────
# 4. EVALUATE ON TEST SET
# ─────────────────────────────────────────
def evaluate_arx(model, test: pd.DataFrame, persistence: dict) -> dict:
    X_test, y_test, idx = build_arx_features(test)
    y_pred = np.clip(model.predict(X_test), 0, 1)

    rmse  = np.sqrt(mean_squared_error(y_test, y_pred))
    mae   = mean_absolute_error(y_test, y_pred)
    skill = 1 - (rmse / persistence["rmse"])

    print(f"\n  ── ARX Test Results ─────────────────────────")
    print(f"  RMSE        : {rmse:.4f}")
    print(f"  MAE         : {mae:.4f}")
    print(f"  Skill Score : {skill:.3f}  (>0 = better than persistence)")
    print(f"  ─────────────────────────────────────────────\n")

    return {"y_true": y_test, "y_pred": y_pred, "idx": idx,
            "rmse": rmse, "mae": mae, "skill": skill}


# ─────────────────────────────────────────
# 5. PLOTS
# ─────────────────────────────────────────
def plot_forecast(results: dict, title: str, n_days: int = 7):
    n   = n_days * 24
    idx = results["idx"][:n]
    y_t = results["y_true"][:n]
    y_p = results["y_pred"][:n]

    fig, axes = plt.subplots(2, 1, figsize=(14, 7))
    fig.suptitle(f"ARX — {title}", fontsize=13, fontweight="bold")

    axes[0].plot(idx, y_t, label="Actual",       color="#264653", linewidth=1.5)
    axes[0].plot(idx, y_p, label="ARX Forecast", color="#E76F51", linewidth=1.5, linestyle="--")
    axes[0].set_ylabel("Normalised Solar Power")
    axes[0].set_title(f"First {n_days} days")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    residuals = y_t - y_p
    axes[1].fill_between(idx, residuals, alpha=0.5, color="#2A9D8F")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Residual")
    axes[1].set_title("Forecast Residuals")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "arx_forecast.png")
    plt.savefig(path, dpi=150)
    print(f"  Plot saved → {path}")
    plt.show()


def plot_comparison(arx: dict, persistence: dict):
    models = ["Persistence\n(baseline)", "ARX"]
    rmses  = [persistence["rmse"], arx["rmse"]]
    maes   = [persistence["mae"],  arx["mae"]]
    x      = np.arange(2)
    width  = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - width/2, rmses, width, label="RMSE", color="#E76F51")
    b2 = ax.bar(x + width/2, maes,  width, label="MAE",  color="#2A9D8F")
    ax.set_title("ARX vs Persistence — Forecast Error (Test Set)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Error (normalised)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    for b in [*b1, *b2]:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.001,
                f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "arx_vs_persistence.png")
    plt.savefig(path, dpi=150)
    print(f"  Plot saved → {path}")
    plt.show()


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("\n=== CE 295 — Step 2a: ARX Baseline Model ===\n")

    # 1. Load splits
    print("[1/4] Loading train / val / test splits...")
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), index_col=0, parse_dates=True)
    val   = pd.read_csv(os.path.join(DATA_DIR, "val.csv"),   index_col=0, parse_dates=True)
    test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"),  index_col=0, parse_dates=True)
    print(f"  Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,} rows")

    # 2. Persistence baseline
    print("\n[2/4] Persistence baseline...")
    persistence = persistence_baseline(test)

    # 3. Train ARX
    print("\n[3/4] Training ARX model...")
    model, val_results = train_arx(train, val)

    # 4. Evaluate on test
    print("[4/4] Evaluating on test set...")
    test_results = evaluate_arx(model, test, persistence)

    # Save predictions
    pred_path = os.path.join(OUTPUT_DIR, "arx_predictions.csv")
    pd.DataFrame({
        "actual":   test_results["y_true"],
        "arx_pred": test_results["y_pred"],
    }, index=test_results["idx"]).to_csv(pred_path)
    print(f"  Predictions saved → {pred_path}")

    # Plots
    plot_forecast(test_results, title="Solar Power Forecast vs Actual (Test Set)")
    plot_comparison(
        {"rmse": test_results["rmse"], "mae": test_results["mae"]},
        persistence
    )

    print("\n✓ ARX baseline complete.")
    print("  Next step → run: lstm_model.py\n")