# hybrid_directional_oil.py
# Two-stage, direction-first hybrid for crude/gas price forecasting
# Stage A (CLS): Direction classifier with threshold tuning to maximize success ratio
# Stage B (REG): Hybrid AR(1–3)+XGB residual for magnitude
# Walk-forward CV (expanding), purge/embargo, volatility scaling, NaN-safe with feature pruning

import os, math, sys, time
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score

# ---------------- CONFIG ----------------
DATA = "data/processed/merged.csv"   # must have 'date' and TARGET_COL (default 'wti_spot')
OUT_DIR = "outputs"

TARGET_COL = "wti_spot"
H_TEST   = 30
MIN_TRAIN= 500
VAL_WIN  = 60
STEP     = 20
PURGE    = 5     # gap before validation to avoid leakage
EMBARGO  = 2     # gap after validation when sliding folds

VOL_SCALE = True
VOL_WIN   = 21
VOL_MINP  = max(2, VOL_WIN//2)
EPS       = 1e-8

# Keep grids small first; expand after it runs
CLS_PARAM_GRID = [
    dict(max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.6, reg_lambda=2.0),
    dict(max_depth=4, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7, reg_lambda=2.0),
]
REG_PARAM_GRID = [
    dict(max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.6, reg_lambda=2.0),
    dict(max_depth=4, learning_rate=0.03, subsample=0.7, colsample_bytree=0.7, reg_lambda=3.0),
]

N_ESTIMATORS = 2000
EARLY_STOP   = 150
TREE_METHOD  = "hist"       # set to "gpu_hist" if you have a CUDA GPU
N_JOBS       = 0
RANDOM_STATE = 32
# ----------------------------------------

def log(msg: str):
    print(time.strftime("[%H:%M:%S]"), msg, flush=True)

def success_ratio(y_true_updown: np.ndarray, y_pred_updown: np.ndarray) -> float:
    mask = np.isfinite(y_true_updown) & np.isfinite(y_pred_updown)
    if not np.any(mask): return float("nan")
    return float((y_true_updown[mask] == y_pred_updown[mask]).mean())

def dir_acc_levels(y_true_levels: np.ndarray, y_pred_levels: np.ndarray) -> float:
    dy_true = np.sign(y_true_levels[1:] - y_true_levels[:-1])
    dy_pred = np.sign(y_pred_levels[1:] - y_pred_levels[:-1])
    if len(dy_true) == 0: return float("nan")
    return float((dy_true == dy_pred).mean())

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Core returns
    out["ret_1"] = out[TARGET_COL].pct_change(1)
    for L in [2,3,5,7,10,21]:
        out[f"ret_{L}"] = out[TARGET_COL].pct_change(L)
    # Level lags
    for l in [1,2,3,5,7,14,21,28,42,60]:
        out[f"lvl_lag{l}"] = out[TARGET_COL].shift(l)
    # Return lags for AR baseline
    for l in [1,2,3]:
        out[f"ret_lag{l}"] = out["ret_1"].shift(l)
    # Rolling vol of returns (as features)
    out["roll_std_21"] = out["ret_1"].rolling(21, min_periods=10).std()
    out["roll_std_63"] = out["ret_1"].rolling(63, min_periods=20).std()
    # Calendar
    out["dow"] = out.index.dayofweek
    out["month"] = out.index.month
    # Macro & curve (add only lagged versions to avoid leakage)
    extra = [
        "crude_inv","wti_prompt_spread","wti_roll_ann",
        "ng_prompt_spread","ng_roll_ann",
        "yield_spread_10_2","oecd_yoy","payems_yoy","wip_yoy",
    ]
    for c in extra:
        if c in out.columns:
            out[f"{c}_lag1"] = out[c].shift(1)
            out[f"{c}_lag3"] = out[c].shift(3)
    # Simple interaction (term structure x inventories) if both exist
    if {"crude_inv","wti_prompt_spread"}.issubset(out.columns):
        out["inv_x_prompt"] = out["crude_inv"] * out["wti_prompt_spread"]
    return out

def compute_global_vol(ret: pd.Series) -> pd.Series:
    vol = ret.rolling(VOL_WIN, min_periods=VOL_MINP).std()
    vol = vol.replace([np.inf,-np.inf], np.nan).ffill().bfill()
    vol = vol.fillna(vol.median()).clip(lower=EPS)
    return vol

# ---------------- SAFE FEATURE SELECTION & PRUNING ----------------

def _select_feature_columns(feat: pd.DataFrame) -> list:
    """
    Keep only lagged/safe features:
      - lvl_lag*, ret_*, roll_std_*
      - *_lag1, *_lag3 (macro/curve lags)
      - dow, month, inv_x_prompt
    Avoid raw contemporaneous macro columns to prevent leakage and NaN bombs.
    """
    keep = []
    for c in feat.columns:
        if c.startswith(("lvl_lag", "ret_", "roll_std_")):
            keep.append(c)
        elif c.endswith(("_lag1", "_lag3")):
            keep.append(c)
        elif c in ("dow", "month", "inv_x_prompt"):
            keep.append(c)
    keep = [c for c in sorted(set(keep)) if c not in {TARGET_COL, "ret_1"}]
    return keep

def _prune_bad_columns(X: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Drop columns that are all-NaN or constant."""
    all_nan = [c for c in X.columns if X[c].isna().all()]
    if all_nan:
        if verbose:
            print(f"[PRUNE] Dropping {len(all_nan)} all-NaN columns:", all_nan[:10], "..." if len(all_nan)>10 else "")
        X = X.drop(columns=all_nan)

    const_cols = []
    for c in X.columns:
        col = X[c].dropna()
        if col.empty:
            continue
        if col.min() == col.max():
            const_cols.append(c)
    if const_cols:
        if verbose:
            print(f"[PRUNE] Dropping {len(const_cols)} constant columns:", const_cols[:10], "..." if len(const_cols)>10 else "")
        X = X.drop(columns=const_cols)
    return X

def build_matrices(feat: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Assemble feature matrix X and targets (y_ret, y_up) with NaN/constant-safe pruning.
    """
    cols = _select_feature_columns(feat)
    if not cols:
        raise RuntimeError("No feature columns selected — check your CSV and feature generation.")

    X = feat[cols].copy()
    X = _prune_bad_columns(X, verbose=True)

    y_ret = feat["ret_1"]
    y_up  = (feat["ret_1"] > 0).astype(int)

    data = pd.concat([y_ret, y_up, X], axis=1)
    data = data.dropna(how="any")
    if len(data) == 0:
        na_share = feat[cols].isna().mean().sort_values(ascending=False)
        print("[DEBUG] Top 20 columns by NaN share BEFORE dropna:\n", na_share.head(20))
        raise RuntimeError(
            "After aligning features/targets, all rows are NaN. "
            "Likely one or more feature columns are entirely NaN. "
            "See the debug NaN shares above and remove/repair those columns."
        )

    y_ret = data.iloc[:, 0]
    y_up  = data.iloc[:, 1]
    X     = data.iloc[:, 2:].astype(float)
    return X, y_ret, y_up

def standardize(X: pd.DataFrame):
    sc = StandardScaler()
    Xs = pd.DataFrame(sc.fit_transform(X), index=X.index, columns=X.columns)
    return Xs, sc

# ----------------- CV UTILITIES -----------------

def rolling_folds(index: pd.Index, h_test: int, min_train: int, val_win: int, step: int, purge: int, embargo: int):
    cut = len(index) - h_test
    idx = index[:cut]
    folds = []
    start = min_train
    while start + val_win <= len(idx):
        tr_end = start - purge
        if tr_end <= 0:
            start += step; continue
        tr = idx[:tr_end]
        va = idx[start:start+val_win]
        folds.append((tr, va))
        start += step + embargo
    return folds

# ----------------- MODELS -----------------

def fit_cls_threshold_tuned(Xs, y_up, tr_idx, va_idx, params):
    model = XGBClassifier(
        n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=N_JOBS,
        tree_method=TREE_METHOD, eval_metric="logloss", **params
    )
    try:
        cb = xgb.callback.EarlyStopping(rounds=EARLY_STOP, save_best=True)
        model.fit(Xs.loc[tr_idx], y_up.loc[tr_idx],
                  eval_set=[(Xs.loc[va_idx], y_up.loc[va_idx])],
                  callbacks=[cb], verbose=False)
    except Exception:
        model.fit(Xs.loc[tr_idx], y_up.loc[tr_idx], verbose=False)

    proba = model.predict_proba(Xs.loc[va_idx])[:,1]
    best_thr, best_sr = 0.5, -1.0
    for thr in np.linspace(0.3, 0.7, 21):
        sr = success_ratio(y_up.loc[va_idx].values, (proba >= thr).astype(int))
        if sr > best_sr:
            best_sr, best_thr = sr, float(thr)
    return model, best_thr, best_sr

def ar_baseline_fit_predict(y_series: pd.Series, tr_idx: pd.Index, pred_idx: pd.Index) -> np.ndarray:
    # AR(1–3) OLS on provided series (can be vol-scaled returns)
    df = pd.DataFrame({
        "y": y_series, "lag1": y_series.shift(1), "lag2": y_series.shift(2), "lag3": y_series.shift(3),
    }).dropna()
    tr = df.loc[df.index.intersection(tr_idx)]
    if len(tr) < 30: return np.zeros(len(pred_idx))
    Xtr = tr[["lag1","lag2","lag3"]].values; ytr = tr["y"].values
    XtX = Xtr.T @ Xtr; Xty = Xtr.T @ ytr
    coef = np.linalg.pinv(XtX) @ Xty
    intercept = float(np.mean(ytr) - np.mean(Xtr, axis=0) @ coef)
    out = []
    for t in pred_idx:
        lags = np.array([y_series.shift(1).loc[t], y_series.shift(2).loc[t], y_series.shift(3).loc[t]], float)
        out.append(0.0 if np.any(pd.isna(lags)) else float(intercept + lags @ coef))
    return np.array(out)

def fit_reg_hybrid(Xs, y_ret, vol, tr_idx, va_idx, params):
    y_model = (y_ret / vol) if VOL_SCALE else y_ret
    ar_va = ar_baseline_fit_predict(y_model, tr_idx, va_idx)
    ar_tr = ar_baseline_fit_predict(y_model, tr_idx, tr_idx)
    res_tr = (y_model.loc[tr_idx].values - ar_tr)

    model = XGBRegressor(
        n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=N_JOBS,
        tree_method=TREE_METHOD, objective="reg:squarederror", **params
    )
    try:
        cb = xgb.callback.EarlyStopping(rounds=EARLY_STOP, save_best=True)
        tail = tr_idx[-min(VAL_WIN, len(tr_idx)):]
        model.fit(Xs.loc[tr_idx], res_tr, eval_set=[(Xs.loc[tail], np.zeros(len(tail)))], callbacks=[cb], verbose=False)
    except Exception:
        model.fit(Xs.loc[tr_idx], res_tr, verbose=False)

    def predict(idx):
        ar_p = ar_baseline_fit_predict(y_model, tr_idx, idx)
        res_p = model.predict(Xs.loc[idx])
        y_hat = ar_p + res_p
        if VOL_SCALE: y_hat = y_hat * vol.loc[idx].values
        return y_hat

    y_hat_va = predict(va_idx)
    return model, predict, y_hat_va

# ----------------- MAIN -----------------

def main():
    log(f"Python: {sys.version.split()[0]} | CWD: {os.getcwd()}")
    log(f"Checking data path: {DATA}")
    if not os.path.exists(DATA):
        log("ERROR: merged.csv not found at the path above.")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    log(f"Outputs will be saved to: {os.path.abspath(OUT_DIR)}")

    df = pd.read_csv(DATA, index_col="date", parse_dates=True).sort_index().ffill()
    log(f"Loaded df with shape {df.shape} and columns: {list(df.columns)[:6]}... (+{len(df.columns)-6} more)")
    if TARGET_COL not in df.columns:
        log(f"ERROR: '{TARGET_COL}' not in CSV columns.")
        return

    # Feature engineering
    feat = make_features(df)
    feat["vol_target"] = compute_global_vol(feat["ret_1"])

    # Build matrices (safe selection + pruning)
    X, y_ret, y_up = build_matrices(feat)
    Xs, _ = standardize(X)
    if Xs.shape[0] == 0:
        raise RuntimeError("No usable rows after pruning/standardizing. Check earlier logs for dropped columns.")

    n_rows, n_feat = Xs.shape
    log(f"Rows after feature dropna: {n_rows} | Features: {n_feat} | H_TEST={H_TEST}")
    if n_rows <= (MIN_TRAIN + VAL_WIN + H_TEST):
        log("ERROR: Not enough rows for MIN_TRAIN + VAL_WIN + H_TEST. "
            f"Need > {MIN_TRAIN + VAL_WIN + H_TEST}, have {n_rows}.")
        return

    idx = Xs.index
    folds = rolling_folds(idx, H_TEST, MIN_TRAIN, VAL_WIN, STEP, PURGE, EMBARGO)
    log(f"CV folds: {len(folds)} (MIN_TRAIN={MIN_TRAIN}, VAL_WIN={VAL_WIN}, STEP={STEP}, PURGE={PURGE}, EMBARGO={EMBARGO})")
    if len(folds) == 0:
        log("ERROR: No CV folds produced. Relax MIN_TRAIN/VAL_WIN/STEP.")
        return

    log(f"Param grids -> CLS:{len(CLS_PARAM_GRID)} | REG:{len(REG_PARAM_GRID)} | Trees:{N_ESTIMATORS} | ES:{EARLY_STOP}")

    # ---- Tune classifier (direction) ----
    best_cls_params, best_thr, best_sr_mean = None, 0.5, -1.0
    for i, pg in enumerate(CLS_PARAM_GRID, 1):
        srs = []
        for j, (tr_idx, va_idx) in enumerate(folds, 1):
            log(f"[CLS] grid {i}/{len(CLS_PARAM_GRID)} fold {j}/{len(folds)}")
            _, thr, sr = fit_cls_threshold_tuned(Xs, y_up, tr_idx, va_idx, pg)
            srs.append(sr)
        sr_mean = float(np.mean(srs))
        log(f"[CLS] params={pg} | mean SR={sr_mean:.3f}")
        if sr_mean > best_sr_mean:
            best_sr_mean = sr_mean
            best_cls_params = pg
            best_thr = float(np.median([x for x in srs if np.isfinite(x)]) if len(srs) else 0.5)
    log(f"[CLS] BEST params={best_cls_params} | SR(mean)={best_sr_mean:.3f} | threshold≈{best_thr:.3f}")

    # ---- Tune regressor (magnitude) ----
    vol_full = feat["vol_target"]
    best_reg_params, best_rmse = None, float("inf")
    for i, pg in enumerate(REG_PARAM_GRID, 1):
        rmses = []
        for j, (tr_idx, va_idx) in enumerate(folds, 1):
            log(f"[REG] grid {i}/{len(REG_PARAM_GRID)} fold {j}/{len(folds)}")
            _, _, y_hat_va = fit_reg_hybrid(Xs, y_ret, vol_full, tr_idx, va_idx, pg)
            y_true_va = y_ret.loc[va_idx].values
            mask = np.isfinite(y_true_va) & np.isfinite(y_hat_va)
            if np.any(mask):
                rmses.append(math.sqrt(mean_squared_error(y_true_va[mask], y_hat_va[mask])))
        rmse_mean = float(np.mean(rmses)) if len(rmses) else np.inf
        log(f"[REG] params={pg} | mean RMSE={rmse_mean:.4f}")
        if rmse_mean < best_rmse:
            best_rmse = rmse_mean
            best_reg_params = pg
    log(f"[REG] BEST params={best_reg_params} | RMSE(mean)={best_rmse:.4f}")

    # ---- Final fit and test ----
    tr_idx = idx[:-H_TEST]
    te_idx = idx[-H_TEST:]
    log(f"Final train rows: {len(tr_idx)} | Test rows: {len(te_idx)}")

    # Stage A: classifier
    log("Fitting final classifier...")
    cls = XGBClassifier(
        n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=N_JOBS,
        tree_method=TREE_METHOD, eval_metric="logloss", **best_cls_params
    )
    try:
        cb = xgb.callback.EarlyStopping(rounds=EARLY_STOP, save_best=True)
        tail = tr_idx[-min(VAL_WIN, len(tr_idx)):]
        cls.fit(Xs.loc[tr_idx], y_up.loc[tr_idx], eval_set=[(Xs.loc[tail], y_up.loc[tail])], callbacks=[cb], verbose=False)
    except Exception:
        cls.fit(Xs.loc[tr_idx], y_up.loc[tr_idx], verbose=False)

    proba_te = cls.predict_proba(Xs.loc[te_idx])[:,1]
    pred_up_te = (proba_te >= best_thr).astype(int)
    sr_te = success_ratio((y_ret.loc[te_idx] > 0).astype(int).values, pred_up_te)
    log(f"[TEST] Direction success ratio = {sr_te:.3f}")

    # Stage B: regressor
    log("Fitting final regressor...")
    reg, reg_predict, _ = fit_reg_hybrid(Xs, y_ret, vol_full, tr_idx, tr_idx[-min(VAL_WIN, len(tr_idx)):], best_reg_params)
    mag_hat_te = np.abs(reg_predict(te_idx))
    ret_hat_te = np.where(pred_up_te == 1, +mag_hat_te, -mag_hat_te)

    # Build price path
    spot_start = df.loc[te_idx[0], TARGET_COL]
    spot_pred = [spot_start]
    for r in ret_hat_te:
        spot_pred.append(spot_pred[-1] * (1.0 + r))
    spot_pred = np.array(spot_pred[1:])
    actual_spot = df.loc[te_idx, TARGET_COL].values

    rmse = float(np.sqrt(mean_squared_error(actual_spot, spot_pred)))
    mae  = float(mean_absolute_error(actual_spot, spot_pred))
    r2   = float(r2_score(actual_spot, spot_pred))
    dacc = dir_acc_levels(actual_spot, spot_pred)

    out_csv = os.path.join(OUT_DIR, "hybrid_directional_forecast.csv")
    pd.DataFrame({
        "date": te_idx, "proba_up": proba_te, "pred_up": pred_up_te,
        "ret_hat": ret_hat_te, "forecast": spot_pred, "actual": actual_spot
    }).to_csv(out_csv, index=False)
    log(f"[SAVE] Forecast CSV -> {os.path.abspath(out_csv)}")

    out_png = os.path.join(OUT_DIR, "hybrid_directional_forecast.png")
    plt.figure(figsize=(12,6))
    plt.plot(te_idx, actual_spot, label="Actual", linewidth=1.3)
    plt.plot(te_idx, spot_pred, label="Hybrid Directional Forecast", linewidth=2)
    plt.title(f"Hybrid Directional 30d | SR={sr_te:.0%} RMSE={rmse:.2f} MAE={mae:.2f} DirAcc(level)={dacc:.0%}")
    plt.legend()
    plt.savefig(out_png, bbox_inches="tight", dpi=150)
    log(f"[SAVE] Chart -> {os.path.abspath(out_png)}")

    log(f"[DONE] SR={sr_te:.0%} RMSE={rmse:.2f} MAE={mae:.2f} R2={r2:.2f} DirAcc(level)={dacc:.0%}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        log("FATAL ERROR:")
        traceback.print_exc()
