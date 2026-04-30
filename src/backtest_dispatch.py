"""
CE 295 - Forecast-to-BESS Backtesting

Run from project root:
    python src/backtest_dispatch.py

Expected inputs:
    data/arx_predictions.csv
    data/lstm_predictions.csv

Outputs saved to:
    results/
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Allow import from src/optimization
sys.path.append(os.path.dirname(__file__))
from optimization.bess_scheduler import BESSOptimizer


# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SOLAR_CAPACITY_KW = 100
PLOT_DAYS = 7

COLORS = {
    "actual": "#264653",
    "arx": "#E76F51",
    "lstm": "#2A9D8F",
    "persistence": "#8D99AE",
    "load": "#457B9D",
    "price": "#F4A261",
    "charge": "#2A9D8F",
    "discharge": "#E76F51",
    "soc": "#264653",
    "grid_no_bess": "#8D99AE",
    "grid_bess": "#2A9D8F",
}


# ─────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────
def savefig(name: str):
    path = os.path.join(RESULTS_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print(f"  Plot saved → {path}")
    plt.show()


def make_price_profile(index):
    hours = index.hour
    return np.where((hours >= 16) & (hours <= 21), 0.45, 0.15)


def make_load_profile(index):
    hours = index.hour
    return 40 + 20 * np.sin(np.pi * (hours - 12) / 12) ** 2


def run_dispatch(name, forecast, load, price, optimizer):
    schedule, cost = optimizer.solve(
        solar_forecast=forecast,
        load_profile=load,
        tou_prices=price,
    )
    schedule["model"] = name
    return schedule, cost


def compute_actual_grid(df, schedule):
    return np.maximum(
        0,
        df["load_kw"].values
        - df["solar_actual_kw"].values
        - schedule["p_dis_kw"].values
        + schedule["p_ch_kw"].values,
    )


# ─────────────────────────────────────────
# CLEAN PLOTS
# ─────────────────────────────────────────
def plot_forecast_comparison(df):
    n = PLOT_DAYS * 24
    plot_df = df.iloc[:n]

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle("Solar Forecast Comparison — Actual vs Forecast Models", fontsize=13, fontweight="bold")

    axes[0].plot(plot_df.index, plot_df["solar_actual_kw"], label="Actual Solar", color=COLORS["actual"], linewidth=1.8)
    axes[0].plot(plot_df.index, plot_df["solar_arx_kw"], label="ARX Forecast", color=COLORS["arx"], linewidth=1.5, linestyle="--")
    axes[0].plot(plot_df.index, plot_df["solar_lstm_kw"], label="LSTM Forecast", color=COLORS["lstm"], linewidth=1.5, linestyle="--")
    axes[0].set_ylabel("Solar Power [kW]")
    axes[0].set_title(f"First {PLOT_DAYS} Days of Test Set")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    arx_resid = plot_df["solar_actual_kw"] - plot_df["solar_arx_kw"]
    lstm_resid = plot_df["solar_actual_kw"] - plot_df["solar_lstm_kw"]
    axes[1].plot(plot_df.index, arx_resid, label="ARX Residual", color=COLORS["arx"], linewidth=1.2)
    axes[1].plot(plot_df.index, lstm_resid, label="LSTM Residual", color=COLORS["lstm"], linewidth=1.2)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].fill_between(plot_df.index, arx_resid, 0, color=COLORS["arx"], alpha=0.12)
    axes[1].fill_between(plot_df.index, lstm_resid, 0, color=COLORS["lstm"], alpha=0.12)
    axes[1].set_ylabel("Residual [kW]")
    axes[1].set_title("Forecast Residuals")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    savefig("clean_solar_forecast_comparison.png")


def plot_battery_dispatch(schedule, model_name="LSTM"):
    n = PLOT_DAYS * 24
    plot_sched = schedule.iloc[:n]

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle(f"BESS Dispatch Behavior — {model_name} Forecast", fontsize=13, fontweight="bold")

    axes[0].plot(plot_sched.index, plot_sched["soc_kwh"], color=COLORS["soc"], linewidth=1.8)
    axes[0].fill_between(plot_sched.index, plot_sched["soc_kwh"], alpha=0.15, color=COLORS["soc"])
    axes[0].set_ylabel("SOC [kWh]")
    axes[0].set_title("Battery State of Charge")
    axes[0].grid(alpha=0.3)

    axes[1].plot(plot_sched.index, plot_sched["p_ch_kw"], label="Charge", color=COLORS["charge"], linewidth=1.5)
    axes[1].plot(plot_sched.index, -plot_sched["p_dis_kw"], label="Discharge", color=COLORS["discharge"], linewidth=1.5)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].fill_between(plot_sched.index, plot_sched["p_ch_kw"], 0, color=COLORS["charge"], alpha=0.18)
    axes[1].fill_between(plot_sched.index, -plot_sched["p_dis_kw"], 0, color=COLORS["discharge"], alpha=0.18)
    axes[1].set_ylabel("Power [kW]")
    axes[1].set_title("Charge/Discharge Power")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    savefig("clean_bess_dispatch_behavior.png")


def plot_grid_import(df, schedule, model_name="LSTM"):
    n = PLOT_DAYS * 24
    grid_no_bess = np.maximum(0, df["load_kw"].values - df["solar_actual_kw"].values)
    grid_with_bess = compute_actual_grid(df, schedule)

    idx = df.index[:n]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(idx, grid_no_bess[:n], label="Without BESS", color=COLORS["grid_no_bess"], linewidth=1.8)
    ax.plot(idx, grid_with_bess[:n], label=f"With BESS ({model_name})", color=COLORS["grid_bess"], linewidth=1.8)
    ax.fill_between(idx, grid_no_bess[:n], grid_with_bess[:n], where=grid_no_bess[:n] >= grid_with_bess[:n], alpha=0.18, color=COLORS["grid_bess"], label="Grid Import Reduction")
    ax.set_title("Grid Import Reduction from Forecast-Based Battery Dispatch", fontweight="bold")
    ax.set_ylabel("Grid Import [kW]")
    ax.legend()
    ax.grid(alpha=0.3)

    savefig("clean_grid_import_comparison.png")


def plot_cost_comparison(results_df):
    # Better than basic bar chart: horizontal ranked bar chart with cost reduction labels
    ordered = results_df.sort_values("actual_backtest_cost", ascending=True).copy()

    fig, ax = plt.subplots(figsize=(9, 5))
    y_pos = np.arange(len(ordered))
    bars = ax.barh(y_pos, ordered["actual_backtest_cost"], color=[COLORS["lstm"], COLORS["arx"], COLORS["persistence"]][:len(ordered)])

    ax.set_yticks(y_pos)
    ax.set_yticklabels(ordered.index)
    ax.set_xlabel("Actual Backtested Cost [$]")
    ax.set_title("Forecast-to-BESS Dispatch Cost Comparison", fontweight="bold")
    ax.grid(alpha=0.3, axis="x")

    for i, (bar, (_, row)) in enumerate(zip(bars, ordered.iterrows())):
        cost = row["actual_backtest_cost"]
        reduction = row["cost_reduction_vs_persistence_%"]
        label = f"${cost:,.0f}"
        if abs(reduction) > 1e-9:
            label += f"  ({reduction:+.2f}%)"
        else:
            label += "  (baseline)"
        ax.text(cost + ordered["actual_backtest_cost"].max() * 0.01, bar.get_y() + bar.get_height()/2, label, va="center", fontsize=10)

    savefig("clean_dispatch_cost_comparison.png")


def plot_daily_cost(df, schedules):
    daily_costs = {}
    for name, schedule in schedules.items():
        grid = compute_actual_grid(df, schedule)
        temp = pd.DataFrame({"cost": grid * df["price"].values}, index=df.index)
        daily_costs[name] = temp["cost"].resample("D").sum()

    daily_df = pd.DataFrame(daily_costs).dropna()

    fig, ax = plt.subplots(figsize=(14, 5))
    for name, color_key in [("Persistence", "persistence"), ("ARX", "arx"), ("LSTM", "lstm")]:
        if name in daily_df.columns:
            ax.plot(daily_df.index, daily_df[name], label=name, linewidth=1.8, color=COLORS[color_key])

    ax.set_title("Daily Backtested Electricity Cost by Forecast Strategy", fontweight="bold")
    ax.set_ylabel("Daily Cost [$]")
    ax.legend()
    ax.grid(alpha=0.3)

    savefig("clean_daily_cost_timeseries.png")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def main():
    print("\n=== CE295 Forecast-to-BESS Backtesting ===\n")

    arx_path = os.path.join(DATA_DIR, "arx_predictions.csv")
    lstm_path = os.path.join(DATA_DIR, "lstm_predictions.csv")

    if not os.path.exists(arx_path):
        raise FileNotFoundError(f"Missing ARX predictions: {arx_path}")
    if not os.path.exists(lstm_path):
        raise FileNotFoundError(f"Missing LSTM predictions: {lstm_path}")

    arx = pd.read_csv(arx_path, index_col=0, parse_dates=True)
    lstm = pd.read_csv(lstm_path, index_col=0, parse_dates=True)

    df = arx.join(lstm[["lstm_pred"]], how="inner")
    df = df.rename(columns={
        "actual": "solar_actual",
        "arx_pred": "solar_arx",
        "lstm_pred": "solar_lstm",
    })

    df["solar_persistence"] = df["solar_actual"].shift(1)
    df = df.dropna()

    df["solar_actual_kw"] = df["solar_actual"] * SOLAR_CAPACITY_KW
    df["solar_arx_kw"] = df["solar_arx"] * SOLAR_CAPACITY_KW
    df["solar_lstm_kw"] = df["solar_lstm"] * SOLAR_CAPACITY_KW
    df["solar_persistence_kw"] = df["solar_persistence"] * SOLAR_CAPACITY_KW

    df["load_kw"] = make_load_profile(df.index)
    df["price"] = make_price_profile(df.index)

    bess_config = {
        "capacity": 200.0,
        "p_max": 50.0,
        "eta_ch": 0.95,
        "eta_dis": 0.95,
        "soc_min_pct": 0.20,
        "soc_max_pct": 0.90,
        "soc_initial_pct": 0.50,
    }

    optimizer = BESSOptimizer(bess_config)
    results = {}
    schedules = {}

    for name, col in {
        "Persistence": "solar_persistence_kw",
        "ARX": "solar_arx_kw",
        "LSTM": "solar_lstm_kw",
    }.items():
        schedule, forecast_cost = run_dispatch(
            name=name,
            forecast=df[col].values,
            load=df["load_kw"].values,
            price=df["price"].values,
            optimizer=optimizer,
        )

        schedule.index = df.index[:len(schedule)]
        schedules[name] = schedule

        actual_grid = compute_actual_grid(df, schedule)
        actual_cost = np.sum(actual_grid * df["price"].values)

        results[name] = {
            "forecast_optimized_cost": forecast_cost,
            "actual_backtest_cost": actual_cost,
        }

        schedule.to_csv(os.path.join(RESULTS_DIR, f"{name.lower()}_schedule.csv"))

    results_df = pd.DataFrame(results).T
    results_df["cost_reduction_vs_persistence_%"] = (
        1 - results_df["actual_backtest_cost"] / results_df.loc["Persistence", "actual_backtest_cost"]
    ) * 100

    results_path = os.path.join(RESULTS_DIR, "dispatch_cost_comparison.csv")
    results_df.to_csv(results_path)

    print("\n── Backtest Cost Summary ─────────────────────────")
    print(results_df.round(3))
    print("──────────────────────────────────────────────────\n")
    print(f"  Results saved → {results_path}")

    print("\n[Plotting] Generating clean figures...")
    plot_forecast_comparison(df)
    plot_battery_dispatch(schedules["LSTM"], model_name="LSTM")
    plot_grid_import(df, schedules["LSTM"], model_name="LSTM")
    plot_cost_comparison(results_df)
    plot_daily_cost(df, schedules)

    print("\n✓ Backtesting complete.\n")


if __name__ == "__main__":
    main()
