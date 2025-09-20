# xgb_forecast.py (hybrid + walk-forward, NaN-safe, DA fix)
# Rolling walk-forward CV, Hybrid AR+XGB residuals, optional classification & vol-scaling.
# Fix: directional accuracy in CV computed on returns (same-length arrays).

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from typing import Dict, Tuple, List

import xgboost as xgb
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler

DATA = "data/processed/merged.csv"
OUT_DIR = "outputs"

# ------------------- CONFIG -------------------
H_TEST = 30        # final out-of-sample horizon (days)
VAL_WIN = 60       # per-fold validation window in rolling CV
MIN_TRAIN = 500    # minimum train size to start CV folds
STEP = 20          # step size between folds
RANDOM_STATE = 32

# Task: "reg" = predict returns; "cls" = predict direction (up/down)
TARGET_MODE = "reg"

# Volatility scaling of target (predict standardized returns)
VOL_SCALE = True
VOL_WIN = 21
VOL_MINP = max(2, VOL_WIN // 2)
EPS = 1e-8  # to avoid division by zero
# ------------------------------------------------

def dir_acc_levels(y_true_levels: np.ndarray, y_pred_levels: np.ndarray) -> float:
    """Directional accuracy computed on LEVEL series via first differences."""
    dy_true = np.sign(y_true_levels[1:] - y_true_levels[:-1])
    dy_pred = np.sign(y_pred_levels[1:] - y_pred_levels[:-1])
    if len(dy_true) == 0:
        return float("nan")
    return float((dy_true == dy_pred).mean())

def dir_acc_returns(y_true_ret: np.ndarray, y_pred_ret: np.ndarray) -> float:
    """Directional accuracy computed directly on RETURN series (same length)."""
    mask = np.isfinite(y_true_ret) & np.isfinite(y_pred_ret)
    if not np.any(mask):
        return float("nan")
    s_true = np.sign(y_true_ret[mask])
    s_pred = np.sign(y_pred_ret[mask])
    return float((s_true == s_pred).mean())

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Lags in levels
    for l in [1,2,3,5,7,14,21,28,42,60]:
        out[f"wti_lag{l}"] = out["wti_spot"].shift(l)
    # Returns & rolling stats
    out["ret_1"] = out["wti_spot"].pct_change(1)
    for L in [2,3,5,7,10,21]:
        out[f"ret_{L}"] = out["wti_spot"].pct_change(L)
    out["roll_std_21"] = out["wti_spot"].pct_change().rolling(21, min_periods=10).std()
    out["roll_std_63"] = out["wti_spot"].pct_change().rolling(63, min_periods=20).std()
    # Return lags for AR baseline
    for l in [1,2,3]:
        out[f"ret_lag{l}"] = out["ret_1"].shift(l)
    # Calendar
    out["dow"] = out.index.dayofweek
    out["month"] = out.index.month
    # Interactions if available
    if {"crude_inv", "wti_prompt_spread"}.issubset(out.columns):
        out["inv_x_prompt"] = out["crude_inv"] * out["wti_prompt_spread"]
    return out

def ar_baseline_fit_predict(y_ret: pd.Series, idx_train: pd.Index, idx_pred: pd.Index) -> np.ndarray:
    """AR(1-3) on returns via OLS; one-step-ahead predictions (no recursive updates)."""
    df = pd.DataFrame({
        "y": y_ret,
        "lag1": y_ret.shift(1),
        "lag2": y_ret.shift(2),
        "lag3": y_ret.shift(3),
    }).dropna()

    tr = df.loc[df.index.intersection(idx_train)]
    if len(tr) < 30:
        return np.zeros(len(idx_pred))

    Xtr = tr[["lag1","lag2","lag3"]].values
    ytr = tr["y"].values
    XtX = Xtr.T @ Xtr
    Xty = Xtr.T @ ytr
    coef = np.linalg.pinv(XtX) @ Xty
    intercept = float(np.mean(ytr) - np.mean(Xtr, axis=0) @ coef)

    preds = []
    for t in idx_pred:
        l1 = y_ret.shift(1).loc[t]
        l2 = y_ret.shift(2).loc[t]
        l3 = y_ret.shift(3).loc[t]
        x = np.array([l1, l2, l3], dtype=float)
        if np.any(pd.isna(x)):
            preds.append(0.0)
        else:
            preds.append(float(intercept + x @ coef))
    return np.array(preds)

def build_Xy(feat: pd.DataFrame, target_mode: str) -> Tuple[pd.DataFrame, pd.Series]:
    base_cols = [c for c in [
        "crude_inv","wti_roll_ann","wti_prompt_spread",
        "ng_roll_ann","ng_prompt_spread",
        "yield_spread_10_2","oecd_yoy","payems_yoy","wip_yoy",
    ] if c in feat.columns]

    lag_cols = [c for c in feat.columns if c.startswith(("wti_lag","ret_","roll_std_"))]
    other_cols = ["dow","month"] + (["inv_x_prompt"] if "inv_x_prompt" in feat.columns else [])

    use_cols = base_cols + lag_cols + other_cols
    X = feat[use_cols].copy()

    if target_mode == "reg":
        y = feat["ret_1"]
    else:
        y = (feat["ret_1"] > 0).astype(int)

    df = pd.concat([y, X], axis=1).dropna()
    y = df.iloc[:,0]
    X = df.iloc[:,1:].astype(float)
    return X, y

def standardize_features(X: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    Xs = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)
    return Xs, scaler

def rolling_folds(index: pd.Index, min_train: int, val_win: int, step: int) -> List[Tuple[pd.Index,pd.Index]]:
    cut = len(index) - H_TEST
    idx = index[:cut]
    folds = []
    start = min_train
    while start + val_win <= len(idx):
        tr = idx[:start]
        va = idx[start:start+val_win]
        folds.append((tr, va))
        start += step
    return folds

# Global rolling volatility (NaN-safe) used everywhere
def compute_global_vol(feat: pd.DataFrame) -> pd.Series:
    vol = feat["ret_1"].rolling(VOL_WIN, min_periods=VOL_MINP).std()
    vol = vol.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    vol = vol.fillna(vol.median()).clip(lower=EPS)
    return vol

def score_fold(
    feat: pd.DataFrame,
    Xs: pd.DataFrame,
    y: pd.Series,
    tr_idx: pd.Index,
    va_idx: pd.Index,
    params: Dict,
    target_mode: str,
    vol_scale: bool
) -> float:
    vol_full = feat["vol_target"]  # NaN-safe

    if target_mode == "reg":
        y_ret = feat["ret_1"].copy()
        y_model = (y_ret / vol_full) if vol_scale else y_ret

        # Baseline AR
        ar_val_pred = ar_baseline_fit_predict(y_model, tr_idx, va_idx)
        ar_tr_pred  = ar_baseline_fit_predict(y_model, tr_idx, tr_idx)
        res_tr = (y_model.loc[tr_idx].values - ar_tr_pred)

        model = XGBRegressor(
            n_estimators=3000, objective="reg:squarederror",
            random_state=RANDOM_STATE, tree_method="hist", n_jobs=0, **params
        )
        try:
            cb = xgb.callback.EarlyStopping(rounds=200, save_best=True)
            model.fit(Xs.loc[tr_idx], res_tr,
                      eval_set=[(Xs.loc[va_idx], np.zeros(len(va_idx)))],
                      callbacks=[cb], verbose=False)
        except Exception:
            model.fit(Xs.loc[tr_idx], res_tr, verbose=False)

        res_va = model.predict(Xs.loc[va_idx])
        y_hat_va = ar_val_pred + res_va
        if vol_scale:
            y_hat_va = y_hat_va * vol_full.loc[va_idx].values

        y_true_va = y_ret.loc[va_idx].values
        # Metrics (same-length, finite)
        mask = np.isfinite(y_true_va) & np.isfinite(y_hat_va)
        y_true_va = y_true_va[mask]; y_hat_va = y_hat_va[mask]

        rmse = np.sqrt(mean_squared_error(y_true_va, y_hat_va))
        dacc = dir_acc_returns(y_true_va, y_hat_va)  # <-- FIX: returns DA
        return rmse - 0.1 * dacc  # lower is better

    else:
        # classification
        y_bin = (feat["ret_1"] > 0).astype(int)
        ar_pred = ar_baseline_fit_predict(feat["ret_1"], tr_idx, va_idx)
        base_dir = (ar_pred > 0).astype(int)

        model = XGBClassifier(
            n_estimators=2000, random_state=RANDOM_STATE,
            tree_method="hist", n_jobs=0, eval_metric="logloss", **params
        )
        try:
            cb = xgb.callback.EarlyStopping(rounds=200, save_best=True)
            model.fit(Xs.loc[tr_idx], y_bin.loc[tr_idx],
                      eval_set=[(Xs.loc[va_idx], y_bin.loc[va_idx])],
                      callbacks=[cb], verbose=False)
        except Exception:
            model.fit(Xs.loc[tr_idx], y_bin.loc[tr_idx], verbose=False)

        proba = model.predict_proba(Xs.loc[va_idx])[:,1]
        pred = (proba >= 0.5).astype(int)
        final = ((pred + base_dir) >= 1).astype(int)
        acc = accuracy_score(y_bin.loc[va_idx], final)
        return 1.0 - acc  # lower is better

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # -------- Load & features --------
    df = pd.read_csv(DATA, index_col="date", parse_dates=True).sort_index().ffill()
    if "wti_spot" not in df.columns:
        raise RuntimeError("wti_spot not found in merged.csv")

    feat = make_features(df)
    feat["vol_target"] = compute_global_vol(feat)

    X, y = build_Xy(feat, TARGET_MODE)
    Xs, scaler = standardize_features(X)

    n = len(Xs)
    if n <= (MIN_TRAIN + VAL_WIN + H_TEST):
        raise RuntimeError(f"Not enough rows ({n}) for MIN_TRAIN({MIN_TRAIN}) + VAL({VAL_WIN}) + TEST({H_TEST}).")

    idx = Xs.index

    # -------- Rolling walk-forward CV --------
    folds = rolling_folds(idx, MIN_TRAIN, VAL_WIN, STEP)
    if len(folds) == 0:
        raise RuntimeError("No CV folds produced; reduce MIN_TRAIN or VAL_WIN.")

    PARAM_GRID = [
        dict(max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.6, reg_lambda=2.0),
        dict(max_depth=4, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7, reg_lambda=2.0),
        dict(max_depth=5, learning_rate=0.02, subsample=0.7, colsample_bytree=0.7, reg_lambda=3.0),
    ]

    cv_rows = []
    best_score = np.inf
    best_params = None

    for pg in PARAM_GRID:
        scores = []
        for tr_idx, va_idx in folds:
            s = score_fold(feat, Xs, y, tr_idx, va_idx, pg, TARGET_MODE, VOL_SCALE)
            scores.append(s)
        mean_s = float(np.mean(scores))
        cv_rows.append({"params": pg, "cv_score": mean_s})
        if mean_s < best_score:
            best_score = mean_s
            best_params = pg

    cv_df = pd.DataFrame(cv_rows)
    cv_csv = os.path.join(OUT_DIR, "xgb_cv_summary.csv")
    cv_df.to_csv(cv_csv, index=False)

    # -------- Final fit up to test start --------
    tr_idx = idx[:-(H_TEST)]
    te_idx = idx[-H_TEST:]

    if TARGET_MODE == "reg":
        y_ret = feat["ret_1"].copy()
        vol_full = feat["vol_target"]
        y_model = (y_ret / vol_full) if VOL_SCALE else y_ret

        ar_te_pred = ar_baseline_fit_predict(y_model, tr_idx, te_idx)
        ar_tr_pred = ar_baseline_fit_predict(y_model, tr_idx, tr_idx)
        res_tr = (y_model.loc[tr_idx].values - ar_tr_pred)

        model = XGBRegressor(
            n_estimators=3000, objective="reg:squarederror",
            random_state=RANDOM_STATE, tree_method="hist", n_jobs=0, **best_params
        )
        try:
            cb = xgb.callback.EarlyStopping(rounds=200, save_best=True)
            model.fit(Xs.loc[tr_idx], res_tr,
                      eval_set=[(Xs.loc[tr_idx[-VAL_WIN:]], np.zeros(min(VAL_WIN, len(tr_idx))))],
                      callbacks=[cb], verbose=False)
            used_es = True
        except Exception:
            model.fit(Xs.loc[tr_idx], res_tr, verbose=False)
            used_es = False

        res_te = model.predict(Xs.loc[te_idx])
        y_hat_te = ar_te_pred + res_te
        if VOL_SCALE:
            y_hat_te = y_hat_te * vol_full.loc[te_idx].values

        # Rebuild spot path by compounding returns
        spot_start = df.loc[te_idx[0], "wti_spot"]
        spot_pred = [spot_start]
        for r in y_hat_te:
            spot_pred.append(spot_pred[-1] * (1.0 + r))
        spot_pred = np.array(spot_pred[1:])
        actual_spot = df.loc[te_idx, "wti_spot"].values

        rmse = float(np.sqrt(mean_squared_error(actual_spot, spot_pred)))
        mae  = float(mean_absolute_error(actual_spot, spot_pred))
        r2   = float(r2_score(actual_spot, spot_pred))
        dacc = dir_acc_levels(actual_spot, spot_pred)  # level-based DA for the plot

        out_csv = os.path.join(OUT_DIR, "xgb_30d_forecast.csv")
        pd.DataFrame({"date": te_idx, "forecast": spot_pred, "actual": actual_spot}).to_csv(out_csv, index=False)

        plt.figure(figsize=(12,6))
        plt.plot(te_idx, actual_spot, label="Actual", linewidth=1.3)
        plt.plot(te_idx, spot_pred, label="Forecast (Hybrid XGB)", linewidth=2)
        t_es = "ES:on" if used_es else "ES:off"
        plt.title(f"Hybrid AR+XGB 30d  |  RMSE={rmse:.2f}  MAE={mae:.2f}  DirAcc={dacc:.0%}  ({t_es})")
        plt.legend()
        out_png = os.path.join(OUT_DIR, "xgb_30d_forecast.png")
        plt.savefig(out_png, bbox_inches="tight", dpi=150)

        print("[CV] Results saved ->", cv_csv)
        print("[BEST PARAMS]", best_params)
        print(f"[TEST] RMSE={rmse:.2f}  MAE={mae:.2f}  R2={r2:.2f}  DirAcc={dacc:.0%}")
        print(f"[SAVE] Forecast CSV  -> {out_csv}")
        print(f"[SAVE] Chart (test)  -> {out_png}")

    else:
        # Classification pipeline: predict direction
        y_bin = (feat["ret_1"] > 0).astype(int)

        model = XGBClassifier(
            n_estimators=2000,
            random_state=RANDOM_STATE, tree_method="hist", n_jobs=0,
            eval_metric="logloss", **best_params
        )
        try:
            cb = xgb.callback.EarlyStopping(rounds=200, save_best=True)
            model.fit(Xs.loc[tr_idx], y_bin.loc[tr_idx],
                      eval_set=[(Xs.loc[tr_idx[-VAL_WIN:]], y_bin.loc[tr_idx[-VAL_WIN:]])],
                      callbacks=[cb], verbose=False)
            used_es = True
        except Exception:
            model.fit(Xs.loc[tr_idx], y_bin.loc[tr_idx], verbose=False)
            used_es = False

        proba = model.predict_proba(Xs.loc[te_idx])[:,1]
        pred_up = (proba >= 0.5).astype(int)
        acc = accuracy_score(y_bin.loc[te_idx], pred_up)

        mu = float(np.abs(feat.loc[tr_idx, "ret_1"]).mean())
        ret_series = np.where(pred_up==1, +mu, -mu)
        spot_start = df.loc[te_idx[0], "wti_spot"]
        spot_pred = [spot_start]
        for r in ret_series:
            spot_pred.append(spot_pred[-1] * (1.0 + r))
        spot_pred = np.array(spot_pred[1:])
        actual_spot = df.loc[te_idx, "wti_spot"].values
        dacc = dir_acc_levels(actual_spot, spot_pred)

        out_csv = os.path.join(OUT_DIR, "xgb_30d_forecast.csv")
        pd.DataFrame({"date": te_idx, "forecast": spot_pred, "actual": actual_spot,
                      "pred_up": pred_up, "proba_up": proba}).to_csv(out_csv, index=False)

        plt.figure(figsize=(12,6))
        plt.plot(te_idx, actual_spot, label="Actual", linewidth=1.3)
        plt.plot(te_idx, spot_pred, label="Dir Forecast (XGB-CLS)", linewidth=2)
        t_es = "ES:on" if used_es else "ES:off"
        plt.title(f"XGB Direction 30d  |  Acc={acc:.0%}  DirAcc(level)={dacc:.0%}  ({t_es})")
        plt.legend()
        out_png = os.path.join(OUT_DIR, "xgb_30d_forecast.png")
        plt.savefig(out_png, bbox_inches="tight", dpi=150)

        print("[CV] Results saved ->", cv_csv)
        print("[BEST PARAMS]", best_params)
        print(f"[TEST] Direction Acc={acc:.0%}")
        print(f"[SAVE] Forecast CSV  -> {out_csv}")
        print(f"[SAVE] Chart (test)  -> {out_png}")

    print("Last 5 test rows (forecast CSV):")
    print(pd.read_csv(os.path.join(OUT_DIR, "xgb_30d_forecast.csv")).tail())

if __name__ == "__main__":
    main()
