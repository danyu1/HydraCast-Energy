# tft_forecast.py
# Temporal Fusion Transformer (PyTorch Forecasting + Lightning 2.x)
# - Trains on data/processed/merged.csv
# - Saves 30d P10/P50/P90 forecast CSV + plot
# - Exports a 1d-ahead trading signal CSV
# - Runs on CPU by default to avoid CUDA illegal memory access on some GPUs

import os
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")  # better stack traces if you later switch to GPU

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
torch.set_float32_matmul_precision("high")  # harmless on CPU; helps if you later enable GPU

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

# ----------------- Config -----------------
DATA = "data/processed/merged.csv"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
INPUT_LEN = 180     # encoder (lookback) days
PRED_LEN = 30       # forecast horizon days
BATCH_SIZE = 128
MAX_EPOCHS = 60
LR = 1e-3
HIDDEN_SIZE = 64
ATTN_HEADS = 4
DROPOUT = 0.10
HIDDEN_CONT_SIZE = 32

# Toggle GPU after you confirm CPU run is stable
USE_GPU = False     # set to True to try GPU later (keep precision="32-true")

# Trading signal (for the next day)
THRESH_USD = 0.30          # long if P50-last >= +$0.30; short if <= -$0.30
UNCERTAINTY_CUTOFF = 2.50   # skip if (P90-P10) > 2.50

# Candidate covariates — only used if present in file
COVARS = [
    "crude_inv",
    "wti_roll_ann", "wti_prompt_spread",
    "ng_roll_ann", "ng_prompt_spread",
    "yield_spread_10_2", "oecd_yoy", "payems_yoy",
    "wip_yoy",
]

def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_frame():
    df = pd.read_csv(DATA, index_col="date", parse_dates=True).sort_index().ffill()
    if "wti_spot" not in df.columns:
        raise RuntimeError("wti_spot not found in merged.csv")
    use_cov = [c for c in COVARS if c in df.columns]
    df = df[["wti_spot"] + use_cov].copy()
    df["time_idx"] = np.arange(len(df), dtype=int)
    df["series"] = "wti"
    return df, use_cov

def build_datasets(df: pd.DataFrame, covars):
    max_idx = df["time_idx"].max()
    cutoff = max_idx - PRED_LEN
    known_reals = list(covars) + ["time_idx"]

    ts_train = TimeSeriesDataSet(
        df[lambda x: x.time_idx <= cutoff],
        time_idx="time_idx",
        target="wti_spot",
        group_ids=["series"],
        min_encoder_length=INPUT_LEN // 2,
        max_encoder_length=INPUT_LEN,
        min_prediction_length=1,
        max_prediction_length=PRED_LEN,
        time_varying_known_reals=known_reals,
        time_varying_unknown_reals=["wti_spot"],
        static_categoricals=["series"],
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    ts_valid = TimeSeriesDataSet.from_dataset(ts_train, df, predict=True, stop_randomization=True)

    # Conservative DataLoaders to avoid GPU transfer quirks; also fine on CPU
    train_loader = ts_train.to_dataloader(
        train=True, batch_size=BATCH_SIZE, num_workers=0, pin_memory=False, persistent_workers=False
    )
    valid_loader = ts_valid.to_dataloader(
        train=False, batch_size=BATCH_SIZE, num_workers=0, pin_memory=False, persistent_workers=False
    )
    return ts_train, ts_valid, train_loader, valid_loader

def train_tft(ts_train, train_loader, valid_loader):
    pl.seed_everything(SEED, workers=True)

    model = TemporalFusionTransformer.from_dataset(
        ts_train,
        learning_rate=LR,
        hidden_size=HIDDEN_SIZE,
        attention_head_size=ATTN_HEADS,
        hidden_continuous_size=HIDDEN_CONT_SIZE,
        dropout=DROPOUT,
        loss=QuantileLoss(),                 # probabilistic (P10..P90)
        optimizer="adam",
        reduce_on_plateau_patience=4,
        log_interval=10,
    )

    early = EarlyStopping(monitor="val_loss", mode="min", patience=8)

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator=("gpu" if USE_GPU else "cpu"),
        devices=1,
        gradient_clip_val=0.1,
        callbacks=[early],
        enable_checkpointing=False,
        enable_progress_bar=True,     # tqdm-style bar
        logger=False,                  # keep simple & dependency-free
        precision="32-true",
        deterministic=True,
    )

    trainer.fit(model, train_loader, valid_loader)
    return model

def make_future(df: pd.DataFrame, covars):
    """Create next PRED_LEN rows with carried-forward covariates."""
    last_idx = int(df["time_idx"].iloc[-1])
    last_row = df.iloc[-1]
    rows = []
    for step in range(1, PRED_LEN + 1):
        r = {"series": "wti", "time_idx": last_idx + step, "wti_spot": float(last_row["wti_spot"])}
        for c in covars:
            r[c] = float(last_row[c])
        rows.append(r)
    return pd.DataFrame(rows)

def make_signal(last_price: float, p10: float, p50: float, p90: float) -> float:
    """Simple 1d-ahead signal: +1 long / -1 short / 0 flat with uncertainty filter."""
    width = p90 - p10
    if width > UNCERTAINTY_CUTOFF:
        return 0.0
    delta = p50 - last_price
    if delta >= THRESH_USD:
        return 1.0
    if delta <= -THRESH_USD:
        return -1.0
    return 0.0

def main():
    set_seed()
    warnings.filterwarnings("ignore")

    # 1) data
    df, covars = load_frame()
    ts_train, ts_valid, train_loader, valid_loader = build_datasets(df, covars)

    # 2) train
    model = train_tft(ts_train, train_loader, valid_loader)

    # 3) build future df (carry-forward known covars) & predict quantiles
    future = make_future(df, covars)
    df_all = pd.concat([df, future], ignore_index=True, sort=False)

    pred_ds = TimeSeriesDataSet.from_dataset(ts_train, df_all, predict=True, stop_randomization=True)
    pred_dl = pred_ds.to_dataloader(train=False, batch_size=256, num_workers=0, pin_memory=False)

    # Use raw predictions to be version-proof
    trainer = pl.Trainer(accelerator=("gpu" if USE_GPU else "cpu"), devices=1, logger=False, enable_progress_bar=False)
    raw_batches = trainer.predict(model, dataloaders=pred_dl, return_predictions=True)

    # Collect "prediction" tensors shaped [B, max_prediction_length, n_quantiles]
    preds_list = []
    for batch in raw_batches:
        if isinstance(batch, dict) and "prediction" in batch:
            preds_list.append(batch["prediction"].detach().cpu())
    if not preds_list:
        raise RuntimeError("No predictions found in trainer.predict output.")
    pred_tensor = torch.cat(preds_list, dim=0)  # [N, max_pred_len, n_q]

    # Pull configured quantiles from the loss
    q = model.loss.quantiles
    # Find nearest indices for 0.1, 0.5, 0.9
    def q_idx(target):
        arr = torch.tensor(q, dtype=torch.float32)
        return int(torch.argmin(torch.abs(arr - target)))
    i10, i50, i90 = q_idx(0.1), q_idx(0.5), q_idx(0.9)

    # Last PRED_LEN rows correspond to the future window
    p10 = pred_tensor[-PRED_LEN:, :, i10].numpy().reshape(-1)
    p50 = pred_tensor[-PRED_LEN:, :, i50].numpy().reshape(-1)
    p90 = pred_tensor[-PRED_LEN:, :, i90].numpy().reshape(-1)

    # 4) save forecast csv
    future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=PRED_LEN, freq="D")
    out_fc = os.path.join(OUT_DIR, "tft_30d_forecast.csv")
    pd.DataFrame({"date": future_dates, "p10": p10, "p50": p50, "p90": p90}).to_csv(out_fc, index=False)

    # 5) plot
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["wti_spot"], label="History", lw=1.4)
    plt.plot(future_dates, p50, label="Forecast (P50)", lw=2)
    plt.fill_between(future_dates, p10, p90, alpha=0.28, label="P10–P90")
    plt.title("HydraCast Energy — TFT 30-Day Probabilistic Forecast")
    plt.legend()
    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, "tft_forecast.png")
    plt.savefig(out_png, dpi=150)

    # 6) 1-day-ahead trade signal
    last_px = float(df["wti_spot"].iloc[-1])
    sig = make_signal(last_px, p10[0], p50[0], p90[0])
    out_sig = os.path.join(OUT_DIR, "tft_1d_signal.csv")
    pd.DataFrame({
        "asof_date": [df.index[-1]],
        "next_date": [future_dates[0]],
        "last_price": [last_px],
        "p10": [p10[0]], "p50": [p50[0]], "p90": [p90[0]],
        "signal": [sig],   # -1 short, 0 flat, +1 long
        "rule": [f"thr={THRESH_USD} USD, unc>{UNCERTAINTY_CUTOFF}→flat"],
    }).to_csv(out_sig, index=False)

    print(f"\n[OK] Saved:")
    print(f" - {out_fc}")
    print(f" - {out_png}")
    print(f" - {out_sig}")
    print("Next-day signal:", "LONG" if sig>0 else "SHORT" if sig<0 else "FLAT")

if __name__ == "__main__":
    main()
