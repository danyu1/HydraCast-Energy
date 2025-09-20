# dl_forecast_tcn.py
# Temporal Convolutional Network (non-LSTM) for 30d WTI forecast with tqdm; robust SAME padding (no Chomp).
import os, math, traceback
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import trange

DATA = "data/processed/merged.csv"
OUT_DIR = "outputs"
H = 30               # forecast horizon (days)
INPUT_LEN = 120      # lookback (days)
BATCH_SIZE = 128
EPOCHS = 200
LR = 1e-3
DROPOUT = 0.1
RANDOM_SEED = 42

# Covariates to include if present in merged.csv
COVARS = [
    "crude_inv",
    "wti_roll_ann", "wti_prompt_spread", "wti_contango_flag", "wti_backwardation_flag",
    "ng_roll_ann", "ng_prompt_spread",
    "yield_spread_10_2", "oecd_yoy", "payems_yoy",
]

# ----------------- Utils -----------------
def set_seed(seed=RANDOM_SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)

def directional_accuracy(actual: np.ndarray, pred: np.ndarray) -> float:
    prev = np.roll(actual, 1)
    a_dir = np.sign(actual - prev)[1:]
    p_dir = np.sign(pred - prev)[1:]
    return float((a_dir == p_dir).mean())

# ----------------- Data Prep -----------------
def load_data():
    df = pd.read_csv(DATA, index_col="date", parse_dates=True).sort_index().ffill()
    if "wti_spot" not in df.columns:
        raise RuntimeError("wti_spot not found in merged.csv")
    covars = [c for c in COVARS if c in df.columns]
    return df, covars

def standardize(train_vals, all_vals, eps=1e-8):
    mean = train_vals.mean(axis=0, keepdims=True)
    std  = train_vals.std(axis=0, keepdims=True) + eps
    return (all_vals - mean) / std, (mean, std)

def build_supervised(df, covars):
    """
    Sliding windows:
      X: [N, C, T] where C = 1 (target) + len(covars); T = INPUT_LEN
      y: next-step (one-day-ahead) target (scaled)
    """
    y = df["wti_spot"].values.astype(np.float32)
    Xcov = df[covars].values.astype(np.float32) if covars else None

    # Train/test split (last H for test)
    y_tr = y[:-H]
    y_all = y.reshape(-1, 1)

    # Scale target using train stats
    y_scaled, (ym, ys) = standardize(y_tr.reshape(-1, 1), y_all)
    y_scaled = y_scaled.flatten()

    # Scale covariates using train stats
    if Xcov is not None:
        Xcov_tr = Xcov[:-H]
        Xcov_scaled, (xm, xs) = standardize(Xcov_tr, Xcov)
    else:
        Xcov_scaled, xm, xs = None, None, None

    # Build windows over full span
    X_list, y_list, idx_list = [], [], []
    total = len(y_scaled)
    for t in range(INPUT_LEN, total):
        x_target = y_scaled[t - INPUT_LEN : t]  # [INPUT_LEN]
        if Xcov_scaled is not None:
            x_cov = Xcov_scaled[t - INPUT_LEN : t, :]       # [INPUT_LEN, F]
            x = np.concatenate([x_target.reshape(-1, 1), x_cov], axis=1)  # [INPUT_LEN, 1+F]
        else:
            x = x_target.reshape(-1, 1)
        X_list.append(x.T)           # -> [C, T]
        y_list.append(y_scaled[t])   # scalar
        idx_list.append(t)

    X_np = np.stack(X_list).astype(np.float32)  # [N, C, T]
    y_np = np.array(y_list, dtype=np.float32)   # [N]
    idx_np = np.array(idx_list)

    N = len(y_scaled)
    test_start_t = N - H
    is_test = idx_np >= test_start_t
    is_train = ~is_test

    return {
        "X_all": X_np,
        "y_all": y_np,
        "idx_all": idx_np,
        "is_train": is_train,
        "is_test": is_test,
        "ym": ym, "ys": ys,
        "dates": df.index,
        "y_raw": y,
    }

def make_loader(X, y, batch_size):
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

# ----------------- TCN Model -----------------
class TemporalBlock(nn.Module):
    """
    TCN block using SAME padding (PyTorch 2+) so output time length == input time length.
    Two Conv1d layers + ReLU + Dropout, with residual skip (1x1 if channels differ).
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            padding='same', dilation=dilation
        )
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            padding='same', dilation=dilation
        )
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.drop1(self.relu1(self.conv1(x)))
        out = self.drop2(self.relu2(self.conv2(out)))
        res = x if self.downsample is None else self.downsample(x)
        # SAME padding ensures time dims already match
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, n_channels, hidden=64, levels=4, kernel_size=3, dropout=0.1):
        super().__init__()
        layers = []
        in_ch = n_channels
        for i in range(levels):
            out_ch = hidden
            dilation = 2 ** i
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch
        self.network = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(hidden, 1))
    def forward(self, x):
        z = self.network(x)
        return self.head(z).squeeze(-1)

# ----------------- Train & Forecast -----------------
def train_model(model, train_loader, epochs=EPOCHS, lr=LR, device="cpu"):
    """
    Trains the TCN and shows tqdm progress with average epoch loss.
    Early stopping if loss doesn't improve for 'patience' epochs.
    """
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    best = math.inf
    patience = 20
    bad = 0

    pbar = trange(1, epochs + 1, desc="Training", leave=True)
    for ep in pbar:
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = crit(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())

        avg = float(np.mean(losses)) if losses else np.nan
        pbar.set_postfix({"epoch": ep, "loss": f"{avg:.6f}"})

        if avg + 1e-9 < best:
            best = avg; bad = 0
        else:
            bad += 1
        if bad >= patience:
            pbar.write("Early stopping (no improvement).")
            break

def iterative_forecast(model, data, device="cpu"):
    """
    Predict the last H points iteratively (one-step ahead) using the
    pre-built windows (no leakage; uses existing covariate windows).
    """
    model.eval()
    preds_scaled = []
    with torch.no_grad():
        for i, tflag in enumerate(data["is_test"]):
            if not tflag:
                continue
            xb = torch.from_numpy(data["X_all"][i:i+1]).to(device)
            pred = model(xb).cpu().numpy().reshape(-1)[0]
            preds_scaled.append(pred)
    ym, ys = data["ym"], data["ys"]
    preds = np.array(preds_scaled) * ys.ravel()[0] + ym.ravel()[0]
    return preds

def main():
    try:
        set_seed()
        os.makedirs(OUT_DIR, exist_ok=True)

        df, covars = load_data()
        data = build_supervised(df, covars)

        X_tr = data["X_all"][data["is_train"]]
        y_tr = data["y_all"][data["is_train"]]
        if len(X_tr) == 0:
            raise RuntimeError("Training set is empty. Try reducing INPUT_LEN or H.")

        loader = make_loader(X_tr, y_tr, BATCH_SIZE)

        n_channels = data["X_all"].shape[1]
        model = TCN(n_channels=n_channels, hidden=64, levels=4, kernel_size=3, dropout=DROPOUT)

        device = "cpu"  # change to "cuda" if you have a GPU and torch-cuda installed
        train_model(model, loader, device=device)

        preds = iterative_forecast(model, data, device=device)
        actual = df["wti_spot"].values[-H:]

        rmse = float(np.sqrt(mean_squared_error(actual, preds)))
        mae  = float(mean_absolute_error(actual, preds))
        dir_acc = directional_accuracy(actual, preds)

        # Save CSV
        out_csv = os.path.join(OUT_DIR, "tcn_30d_forecast.csv")
        test_idx = df.index[-H:]
        pd.DataFrame({"date": test_idx, "forecast": preds, "actual": actual}).to_csv(out_csv, index=False)

        # Save plot
        out_png = os.path.join(OUT_DIR, "tcn_30d_forecast.png")
        plt.figure(figsize=(12, 6))
        plt.plot(test_idx, actual, label="Actual", linewidth=1.6)
        plt.plot(test_idx, preds, label="Forecast (TCN)", linewidth=2)
        plt.title(f"HydraCast Energy — TCN 30-day (RMSE={rmse:.2f}, MAE={mae:.2f}, DirAcc={dir_acc:.0%})")
        plt.legend()
        plt.savefig(out_png, bbox_inches="tight", dpi=150)

        print(f"\n[OK] TCN channels={n_channels}, input_len={INPUT_LEN}")
        print(f"[OK] RMSE={rmse:.3f}  MAE={mae:.3f}  DirAcc={dir_acc:.2%}")
        print(f"[OK] Saved chart  -> {out_png}")
        print(f"[OK] Saved CSV    -> {out_csv}")

    except Exception:
        print("[ERROR] dl_forecast_tcn.py crashed:\n")
        traceback.print_exc()

if __name__ == "__main__":
    main()
