"""
CE 295 - Data Science for Energy
Step 2b: LSTM Model for Solar Power Forecasting
Authors: Sohel & François

Architecture:
    Input  : sliding window of 24 hours x 8 features
    LSTM 1 : 64 hidden units + Dropout(0.2)
    LSTM 2 : 32 hidden units + Dropout(0.2)
    Output : 1 unit (solar_power forecast t+1)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
# Resolve project root dynamically (assuming script is in src/models/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed") # Path to train.csv, val.csv, test.csv
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data")            # Path where predictions are saved

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model hyperparameters
WINDOW     = 24       # hours of history fed to LSTM
FEATURES   = ["ghi", "dni", "dhi", "temp", "wind_speed", "humidity", "hour_sin", "hour_cos"]
TARGET     = "solar_power"
HIDDEN1    = 64
HIDDEN2    = 32
DROPOUT    = 0.2
BATCH_SIZE = 32
EPOCHS     = 50
LR         = 1e-3
PATIENCE   = 5        # early stopping patience

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Using device: {DEVICE}")


# ─────────────────────────────────────────
# 1. DATASET
# ─────────────────────────────────────────
class SolarDataset(Dataset):
    """
    Sliding window dataset.
    X[i] : window of shape (WINDOW, n_features)
    y[i] : solar_power at time i + WINDOW
    """
    def __init__(self, df: pd.DataFrame, window: int = WINDOW):
        self.X = []
        self.y = []
        features = df[FEATURES].values
        target   = df[TARGET].values

        for i in range(len(df) - window):
            self.X.append(features[i : i + window])
            self.y.append(target[i + window])

        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────
# 2. MODEL ARCHITECTURE
# ─────────────────────────────────────────
class LSTMForecaster(nn.Module):
    def __init__(self, n_features: int, hidden1: int, hidden2: int, dropout: float):
        super().__init__()
        self.lstm1 = nn.LSTM(n_features, hidden1, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.fc    = nn.Linear(hidden2, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)           # (batch, window, hidden1)
        out     = self.drop1(out)
        out, _ = self.lstm2(out)          # (batch, window, hidden2)
        out     = self.drop2(out)
        out     = out[:, -1, :]           # take last timestep
        out     = self.fc(out)            # (batch, 1)
        return out.squeeze(1)


# ─────────────────────────────────────────
# 3. TRAINING LOOP
# ─────────────────────────────────────────
def train_lstm(train_df: pd.DataFrame, val_df: pd.DataFrame) -> tuple:
    train_ds = SolarDataset(train_df)
    val_ds   = SolarDataset(val_df)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    model     = LSTMForecaster(len(FEATURES), HIDDEN1, HIDDEN2, DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Training for up to {EPOCHS} epochs (early stopping patience={PATIENCE})...\n")

    for epoch in range(1, EPOCHS + 1):
        # ── Train
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_dl:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(y_batch)
        train_loss = epoch_loss / len(train_ds)

        # ── Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_dl:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                pred     = model(X_batch)
                val_loss += criterion(pred, y_batch).item() * len(y_batch)
        val_loss /= len(val_ds)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS} — Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")

        # ── Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch} (best val loss: {best_val_loss:.5f})")
                break

    # Restore best weights
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "lstm_weights.pt"))
    print(f"  Model saved → {OUTPUT_DIR}/lstm_weights.pt")

    return model, train_losses, val_losses


# ─────────────────────────────────────────
# 4. EVALUATION
# ─────────────────────────────────────────
def evaluate_lstm(model, test_df: pd.DataFrame, persistence_rmse: float) -> dict:
    test_ds = SolarDataset(test_df)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_dl:
            X_batch = X_batch.to(DEVICE)
            pred    = model(X_batch).cpu().numpy()
            preds.extend(pred)
            actuals.extend(y_batch.numpy())

    y_pred = np.clip(np.array(preds),   0, 1)
    y_true = np.array(actuals)
    idx    = test_df.index[WINDOW:]

    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    mae   = mean_absolute_error(y_true, y_pred)
    skill = 1 - (rmse / persistence_rmse)

    print(f"\n  ── LSTM Test Results ────────────────────────")
    print(f"  RMSE        : {rmse:.4f}")
    print(f"  MAE         : {mae:.4f}")
    print(f"  Skill Score : {skill:.3f}  (>0 = better than persistence)")
    print(f"  ─────────────────────────────────────────────\n")

    return {"y_true": y_true, "y_pred": y_pred, "idx": idx,
            "rmse": rmse, "mae": mae, "skill": skill}


# ─────────────────────────────────────────
# 5. PLOTS
# ─────────────────────────────────────────
def plot_training_curve(train_losses: list, val_losses: list):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train_losses, label="Train Loss", color="#264653", linewidth=1.5)
    ax.plot(val_losses,   label="Val Loss",   color="#E76F51", linewidth=1.5)
    ax.set_title("LSTM Training Curve", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "lstm_training_curve.png")
    plt.savefig(path, dpi=150)
    print(f"  Plot saved → {path}")
    plt.show()


def plot_forecast(results: dict, n_days: int = 7):
    n   = n_days * 24
    idx = results["idx"][:n]
    y_t = results["y_true"][:n]
    y_p = results["y_pred"][:n]

    fig, axes = plt.subplots(2, 1, figsize=(14, 7))
    fig.suptitle("LSTM — Solar Power Forecast vs Actual (Test Set)", fontsize=13, fontweight="bold")

    axes[0].plot(idx, y_t, label="Actual",        color="#264653", linewidth=1.5)
    axes[0].plot(idx, y_p, label="LSTM Forecast", color="#2A9D8F", linewidth=1.5, linestyle="--")
    axes[0].set_ylabel("Normalised Solar Power")
    axes[0].set_title(f"First {n_days} days of test set")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    residuals = y_t - y_p
    axes[1].fill_between(idx, residuals, alpha=0.5, color="#E76F51")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Residual")
    axes[1].set_title("Forecast Residuals")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "lstm_forecast.png")
    plt.savefig(path, dpi=150)
    print(f"  Plot saved → {path}")
    plt.show()


def plot_final_comparison(persistence_rmse: float, persistence_mae: float,
                          arx_rmse: float,         arx_mae: float,
                          lstm_rmse: float,         lstm_mae: float):
    models = ["Persistence\n(baseline)", "ARX", "LSTM"]
    rmses  = [persistence_rmse, arx_rmse, lstm_rmse]
    maes   = [persistence_mae,  arx_mae,  lstm_mae]
    colors = ["#AAAAAA", "#E76F51", "#2A9D8F"]
    x      = np.arange(3)
    width  = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - width/2, rmses, width, label="RMSE", color=colors, alpha=0.85)
    b2 = ax.bar(x + width/2, maes,  width, label="MAE",  color=colors, alpha=0.55)
    ax.set_title("Persistence vs ARX vs LSTM — Test Set Error", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Error (normalised)")
    ax.legend(["RMSE", "MAE"])
    ax.grid(alpha=0.3, axis="y")
    for b in [*b1, *b2]:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.001,
                f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "model_comparison.png")
    plt.savefig(path, dpi=150)
    print(f"  Plot saved → {path}")
    plt.show()


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("\n=== CE 295 — Step 2b: LSTM Model ===\n")

    # 1. Load data
    print("[1/5] Loading train / val / test splits...")
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), index_col=0, parse_dates=True)
    val   = pd.read_csv(os.path.join(DATA_DIR, "val.csv"),   index_col=0, parse_dates=True)
    test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"),  index_col=0, parse_dates=True)
    print(f"  Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,} rows")

    # 2. Persistence baseline (needed for skill score)
    print("\n[2/5] Computing persistence baseline...")
    y_test  = test[TARGET].values
    p_rmse  = np.sqrt(mean_squared_error(y_test[1:], y_test[:-1]))
    p_mae   = mean_absolute_error(y_test[1:], y_test[:-1])
    print(f"  Persistence — RMSE: {p_rmse:.4f} | MAE: {p_mae:.4f}")

    # 3. Train LSTM
    print("\n[3/5] Training LSTM...")
    model, train_losses, val_losses = train_lstm(train, val)
    plot_training_curve(train_losses, val_losses)

    # 4. Evaluate
    print("[4/5] Evaluating LSTM on test set...")
    results = evaluate_lstm(model, test, p_rmse)

    # Save predictions
    pred_path = os.path.join(OUTPUT_DIR, "lstm_predictions.csv")
    pd.DataFrame({
        "actual":    results["y_true"],
        "lstm_pred": results["y_pred"],
    }, index=results["idx"]).to_csv(pred_path)
    print(f"  Predictions saved → {pred_path}")

    # 5. Plots
    print("[5/5] Generating plots...")
    plot_forecast(results)

    # Load ARX results if available for final comparison
    arx_path = os.path.join(OUTPUT_DIR, "arx_predictions.csv")
    if os.path.exists(arx_path):
        arx_df  = pd.read_csv(arx_path, index_col=0, parse_dates=True)
        arx_rmse = np.sqrt(mean_squared_error(arx_df["actual"], arx_df["arx_pred"]))
        arx_mae  = mean_absolute_error(arx_df["actual"], arx_df["arx_pred"])
        plot_final_comparison(p_rmse, p_mae, arx_rmse, arx_mae, results["rmse"], results["mae"])
    else:
        print("  (Run arx_model.py first to generate the 3-model comparison plot)")

    print("\n✓ LSTM training complete.")
    print("  Next step → run: bess_optimization.py\n")