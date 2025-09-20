# xgb_forecast.py
# Robust XGBoost script: time-series split, callback-based early stopping (fallback-safe),
# directional accuracy, and feature importance.

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

DATA = "data/processed/merged.csv"
OUT_DIR = "outputs"

H_TEST = 30      # final test horizon
H_VAL  = 90      # validation window just before the test window
RANDOM_STATE = 32

def dir_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Directional accuracy of day-to-day changes."""
    dy_true = np.sign(y_true[1:] - y_true[:-1])
    dy_pred = np.sign(y_pred[1:] - y_pred[:-1])
    if len(dy_true) == 0:
        return float("nan")
    return float((dy_true == dy_pred).mean())

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Lags
    for l in [1, 2, 3, 5, 7, 14, 21, 28, 42, 60]:
        out[f"wti_lag{l}"] = out["wti_spot"].shift(l)
    # Returns & rolling stats
    out["wti_ret_7"]  = out["wti_spot"].pct_change(7)
    out["wti_ret_21"] = out["wti_spot"].pct_change(21)
    out["wti_roll_std_21"] = out["wti_spot"].rolling(21, min_periods=10).std()
    out["wti_roll_std_63"] = out["wti_spot"].rolling(63, min_periods=20).std()
    # Calendar
    out["dow"] = out.index.dayofweek
    out["month"] = out.index.month
    # Interaction if available
    if {"crude_inv", "wti_prompt_spread"}.issubset(out.columns):
        out["inv_x_prompt"] = out["crude_inv"] * out["wti_prompt_spread"]
    return out

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # -------- Load --------
    df = pd.read_csv(DATA, index_col="date", parse_dates=True).sort_index().ffill()
    if "wti_spot" not in df.columns:
        raise RuntimeError("wti_spot not found in merged.csv")

    base_cols = [c for c in [
        "crude_inv",
        "wti_roll_ann", "wti_prompt_spread",
        "ng_roll_ann", "ng_prompt_spread",
        "yield_spread_10_2", "oecd_yoy", "payems_yoy", "wip_yoy",
    ] if c in df.columns]

    feat = make_features(df)

    lag_cols   = [c for c in feat.columns if c.startswith(("wti_lag", "wti_ret_", "wti_roll_std_"))]
    other_cols = ["dow", "month"] + (["inv_x_prompt"] if "inv_x_prompt" in feat.columns else [])

    use_cols = base_cols + lag_cols + other_cols
    X = feat[use_cols]
    y = feat["wti_spot"]

    data = pd.concat([y, X], axis=1).dropna()
    y = data.iloc[:, 0].astype(float)
    X = data.iloc[:, 1:].astype(float)

    # -------- Splits --------
    n = len(X)
    if n <= (H_VAL + H_TEST + 50):
        raise RuntimeError(f"Not enough rows ({n}) after feature dropna for VAL({H_VAL}) + TEST({H_TEST}).")

    idx_all  = X.index
    idx_tr   = idx_all[:-(H_VAL + H_TEST)]
    idx_val  = idx_all[-(H_VAL + H_TEST):-H_TEST]
    idx_te   = idx_all[-H_TEST:]

    X_tr, y_tr = X.loc[idx_tr],  y.loc[idx_tr]
    X_val, y_val = X.loc[idx_val], y.loc[idx_val]
    X_te,  y_te  = X.loc[idx_te],  y.loc[idx_te]

    # -------- Model --------
    model = XGBRegressor(
        n_estimators=4000,
        max_depth=7,
        learning_rate=0.02,
        subsample=0.90,
        colsample_bytree=0.90,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=0,
        # (omit eval_metric to be safest across versions)
    )

    used_es = False
    fit_kwargs = dict(
        X=X_tr, y=y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Try callback-based EarlyStopping; if unavailable, train without ES.
    try:
        from xgboost import callback as xgb_cb
        es_cb = xgb_cb.EarlyStopping(rounds=200, save_best=True)
        model.fit(**fit_kwargs, callbacks=[es_cb])
        used_es = True
    except Exception:
        model.fit(**fit_kwargs)

    # -------- Validation metrics --------
    val_pred = model.predict(X_val)
    val_rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))
    val_mae  = float(mean_absolute_error(y_val, val_pred))
    val_r2   = float(r2_score(y_val, val_pred))
    val_dir  = dir_acc(y_val.values, val_pred)

    # -------- Test forecast --------
    pred = model.predict(X_te)
    rmse = float(np.sqrt(mean_squared_error(y_te, pred)))
    mae  = float(mean_absolute_error(y_te, pred))
    r2   = float(r2_score(y_te, pred))
    dacc = dir_acc(y_te.values, pred)

    # -------- Save CSVs --------
    out_csv = os.path.join(OUT_DIR, "xgb_30d_forecast.csv")
    pd.DataFrame({"date": y_te.index, "forecast": pred, "actual": y_te.values}).to_csv(out_csv, index=False)

    out_csv_val = os.path.join(OUT_DIR, "xgb_val_window.csv")
    pd.DataFrame({"date": y_val.index, "forecast": val_pred, "actual": y_val.values}).to_csv(out_csv_val, index=False)

    # -------- Plots --------
    plt.figure(figsize=(12, 6))
    plt.plot(y_te.index, y_te.values, label="Actual", linewidth=1.3)
    plt.plot(y_te.index, pred, label="Forecast (XGB)", linewidth=2)
    title_es = "ES:on" if used_es else "ES:off"
    plt.title(f"HydraCast Energy — XGB 30d  |  RMSE={rmse:.2f}  MAE={mae:.2f}  DirAcc={dacc:.0%}  ({title_es})")
    plt.legend()
    out_png = os.path.join(OUT_DIR, "xgb_30d_forecast.png")
    plt.savefig(out_png, bbox_inches="tight", dpi=150)

    # Feature importance (top 20)
    out_fi = None
    try:
        importances = model.feature_importances_
        fi = pd.Series(importances, index=X.columns).sort_values(ascending=False).head(20)
        plt.figure(figsize=(10, 7))
        fi[::-1].plot(kind="barh")
        plt.title("XGB Feature Importance (top 20)")
        plt.tight_layout()
        out_fi = os.path.join(OUT_DIR, "xgb_feature_importance.png")
        plt.savefig(out_fi, dpi=150)
    except Exception:
        pass

    # -------- Console summary --------
    best_iter = getattr(model, "best_iteration", None)
    best_ntree = (best_iter + 1) if isinstance(best_iter, int) else "N/A"
    print(f"[OK] XGB fitted. Early stopping used: {used_es}. Best trees: {best_ntree}")
    print(f"[OK] Train size: {len(X_tr)} | Val: {len(X_val)} | Test: {len(X_te)}")
    print(f"[VAL] RMSE={val_rmse:.2f}  MAE={val_mae:.2f}  R2={val_r2:.2f}  DirAcc={val_dir:.0%}")
    print(f"[TEST] RMSE={rmse:.2f}  MAE={mae:.2f}  R2={r2:.2f}  DirAcc={dacc:.0%}")
    print(f"[SAVE] Forecast CSV  -> {out_csv}")
    print(f"[SAVE] Val CSV       -> {out_csv_val}")
    print(f"[SAVE] Chart (test)  -> {out_png}")
    if out_fi:
        print(f"[SAVE] Feature importance -> {out_fi}")
    print("Last 5 test rows:")
    print(pd.read_csv(out_csv).tail())

if __name__ == "__main__":
    main()
