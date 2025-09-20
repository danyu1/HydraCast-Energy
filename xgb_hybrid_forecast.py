# xgb_hybrid_returns.py
# AR(1..3) + XGBoost on next-day *residual return* to reduce phase lag.
# Proper target-day alignment, short-horizon momentum, residual AR lags.

import os, sys, traceback
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
from datetime import datetime

DATA = "data/processed/merged.csv"
OUT_DIR = "outputs"

H_TEST   = 30
H_VAL    = 90
BURN_IN  = 3
RANDOM_STATE = 32

def log(msg, fh=None):
    print(msg, flush=True)
    if fh: fh.write(msg + "\n"); fh.flush()

def dir_acc(y_true, y_pred):
    dy_true = np.sign(np.diff(y_true))
    dy_pred = np.sign(np.diff(y_pred))
    return float((dy_true == dy_pred).mean()) if len(dy_true) else float("nan")

def ema(s: pd.Series, span: int, minp=None) -> pd.Series:
    return s.ewm(span=span, min_periods=minp or max(2, span // 3)).mean()

def build_features(df: pd.DataFrame):
    f = df.copy()

    # AR lags for the level model
    for l in (1, 2, 3):
        f[f"wti_lag{l}"] = f["wti_spot"].shift(l)

    # Short-horizon momentum to reduce lag
    f["ret_1"] = f["wti_spot"].diff(1)
    f["ret_3"] = f["wti_spot"].diff(3)
    f["ret_5"] = f["wti_spot"].diff(5)

    # Macro/curve (use only if present), with short lags/EMA to keep them reactive
    macro_raw = [
        "crude_inv", "wti_prompt_spread", "wti_roll_ann",
        "ng_prompt_spread", "ng_roll_ann",
        "yield_spread_10_2", "oecd_yoy", "payems_yoy", "wip_yoy",
    ]
    macro_raw = [c for c in macro_raw if c in f.columns]

    macro_cols = []
    for c in macro_raw:
        f[f"{c}_lag3"]  = f[c].shift(3)
        f[f"{c}_lag7"]  = f[c].shift(7)
        f[f"{c}_ema10"] = ema(f[c], 10)   # quicker than 21
        macro_cols += [f"{c}_lag3", f"{c}_lag7", f"{c}_ema10"]

    if {"crude_inv", "wti_prompt_spread"}.issubset(f.columns):
        f["inv_x_prompt"] = f["crude_inv"] * f["wti_prompt_spread"]
        macro_cols.append("inv_x_prompt")

    f["dow"] = f.index.dayofweek
    f["month"] = f.index.month
    macro_cols += ["dow", "month"]

    return f, macro_cols

def metrics(split, y_true, y_hat, fh=None):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_hat)))
    mae  = float(mean_absolute_error(y_true, y_hat))
    r2   = float(r2_score(y_true, y_hat))
    dacc = dir_acc(y_true, y_hat)
    log(f"[{split}] RMSE={rmse:.3f}  MAE={mae:.3f}  R2={r2:.3f}  DirAcc={dacc:.0%}", fh)
    return rmse, mae, r2, dacc

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    log_path = os.path.join(OUT_DIR, "run_log.txt")
    with open(log_path, "a", encoding="utf-8") as fh:
        log(f"\n===== Hybrid-Returns run @ {datetime.now()} =====", fh)

        # ---------- LOAD ----------
        if not os.path.exists(DATA):
            log(f"[FATAL] Missing {DATA}", fh); sys.exit(1)
        df = pd.read_csv(DATA, index_col="date", parse_dates=True).sort_index().ffill()
        if "wti_spot" not in df.columns:
            log("[FATAL] 'wti_spot' missing", fh); sys.exit(1)
        log(f"[INFO] Loaded {len(df):,} rows, cols={list(df.columns)}", fh)

        # ---------- FEATURES ----------
        feat, macro_cols = build_features(df)
        ar_cols = ["wti_lag1", "wti_lag2", "wti_lag3"]
        need = ["wti_spot"] + ar_cols + macro_cols + ["ret_1", "ret_3", "ret_5"]
        use = feat[need].dropna()
        if len(use) <= (H_VAL + H_TEST + 50):
            log(f"[FATAL] Not enough rows after dropna ({len(use)})", fh); sys.exit(1)

        y = use["wti_spot"].astype(float)
        X_ar  = use[ar_cols].astype(float)
        X_res = use[macro_cols + ["ret_1", "ret_3", "ret_5"]].astype(float)

        idx_all = use.index
        idx_train_target = idx_all[:-(H_VAL + H_TEST)]
        idx_val_target   = idx_all[-(H_VAL + H_TEST):-H_TEST]
        idx_test_target  = idx_all[-H_TEST:]
        log(f"[INFO] Target splits -> train={len(idx_train_target)}, val={len(idx_val_target)}, test={len(idx_test_target)}", fh)

        # ---------- STAGE 1: AR level model ----------
        ar = LinearRegression()
        ar.fit(X_ar.loc[idx_train_target], y.loc[idx_train_target])
        ar_hat = pd.Series(ar.predict(X_ar), index=idx_all)  # âr[t]

        # Residual (level) series for residual lags (known at t)
        res_series = (y - ar_hat)
        X_res["res_lag1"] = res_series.shift(1)
        X_res["res_lag2"] = res_series.shift(2)
        X_res["res_lag3"] = res_series.shift(3)

        # Re-dropna after adding residual lags
        aligned = pd.concat([y, ar_hat.rename("ar_hat"), X_res], axis=1).dropna()
        y = aligned["wti_spot"]
        ar_hat = aligned["ar_hat"]
        X_res = aligned.drop(columns=["wti_spot", "ar_hat"])

        # ---------- NEXT-DAY RETURN TARGET ----------
        # delta_next[t]     = y[t+1] - y[t]
        # ar_delta_next[t]  = âr[t+1] - y[t]
        # resid_delta_next  = delta_next - ar_delta_next
        y_next = y.shift(-1)
        delta_next = y_next - y
        ar_delta_next = ar_hat.shift(-1) - y
        resid_delta_next = delta_next - ar_delta_next

        # keep rows with a valid next-day target
        target_frame = pd.concat(
            [resid_delta_next.rename("resid_delta_next"), X_res], axis=1
        ).dropna()

        feat_idx = target_frame.index                # time t (feature time)
        targ_idx = feat_idx + pd.Timedelta(days=1)   # target day t+1

        # masks for splits by TARGET day
        def mask_for_target(target_slice):
            tset = set(target_slice)
            return target_frame.index[[ti in tset for ti in targ_idx]]

        idx_train_feat = mask_for_target(idx_train_target)
        idx_val_feat   = mask_for_target(idx_val_target)
        idx_test_feat  = mask_for_target(idx_test_target)

        y_res_tr  = target_frame.loc[idx_train_feat, "resid_delta_next"].values
        X_res_tr  = target_frame.loc[idx_train_feat].drop(columns=["resid_delta_next"]).values
        y_res_val = target_frame.loc[idx_val_feat,   "resid_delta_next"].values
        X_res_val = target_frame.loc[idx_val_feat].drop(columns=["resid_delta_next"]).values
        y_res_te  = target_frame.loc[idx_test_feat,  "resid_delta_next"].values
        X_res_te  = target_frame.loc[idx_test_feat].drop(columns=["resid_delta_next"]).values
        feat_names = list(target_frame.drop(columns=["resid_delta_next"]).columns)

        # AR deltas and truth for TARGET days
        y_val_true = y.loc[idx_val_target].values
        y_te_true  = y.loc[idx_test_target].values
        ar_val_delta = (ar_hat.shift(-1) - y).loc[idx_val_feat].values
        ar_te_delta  = (ar_hat.shift(-1) - y).loc[idx_test_feat].values
        y_val_start  = y.loc[idx_val_feat].values
        y_te_start   = y.loc[idx_test_feat].values

        # ---------- STAGE 2: XGB on next-day residual return ----------
        dtrain = xgb.DMatrix(X_res_tr, label=y_res_tr, feature_names=feat_names)
        dval   = xgb.DMatrix(X_res_val, label=y_res_val, feature_names=feat_names)
        dtest  = xgb.DMatrix(X_res_te, feature_names=feat_names)

        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "eta": 0.04,
            "max_depth": 5,
            "subsample": 0.90,
            "colsample_bytree": 0.70,
            "gamma": 0.5,
            "reg_alpha": 0.5,
            "reg_lambda": 1.5,
            "tree_method": "hist",
            "seed": RANDOM_STATE,
        }

        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=4000,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=200,
            verbose_eval=False,
        )
        log(f"[INFO] Best iteration: {booster.best_iteration + 1}", fh)

        res_val_delta_hat = booster.predict(dval)
        res_te_delta_hat  = booster.predict(dtest)

        # Reconstruct level forecasts for TARGET day: y_hat[t+1] = y[t] + ar_delta_next[t] + resid_delta_hat[t]
        y_val_hat = y_val_start + ar_val_delta + res_val_delta_hat
        y_te_hat  = y_te_start  + ar_te_delta  + res_te_delta_hat

        # ---------- METRICS ----------
        metrics("VAL ", y_val_true, y_val_hat, fh)
        rmse, mae, r2, dacc = metrics("TEST", y_te_true, y_te_hat, fh)
        if 0 < BURN_IN < len(y_te_true):
            metrics("TEST (post-burn-in)", y_te_true[BURN_IN:], y_te_hat[BURN_IN:], fh)

        # ---------- SAVE (by TARGET day) ----------
        out_csv = os.path.join(OUT_DIR, "xgb_hybrid_returns_30d.csv")
        pd.DataFrame({
            "date": idx_test_target,
            "forecast": y_te_hat,
            "actual": y_te_true,
            "ar_only_level": y_te_start + ar_te_delta,          # AR next-day level
            "residual_delta_hat": res_te_delta_hat,             # ML next-day residual return
        }).to_csv(out_csv, index=False)
        log(f"[SAVE] Forecast CSV   -> {out_csv}", fh)

        plt.figure(figsize=(12, 6))
        plt.plot(idx_test_target, y_te_true, label="Actual", lw=1.4)
        plt.plot(idx_test_target, y_te_hat,  label="Hybrid (Δ next-day)", lw=2)
        plt.plot(idx_test_target, (y_te_start + ar_te_delta), label="AR-only (Δ next-day)", lw=1, alpha=0.7)
        plt.title(f"Hybrid (returns) 30d | RMSE={rmse:.2f} MAE={mae:.2f} DirAcc={dacc:.0%}")
        plt.legend(); plt.tight_layout()
        out_png = os.path.join(OUT_DIR, "xgb_hybrid_returns_30d.png")
        plt.savefig(out_png, dpi=150)
        log(f"[SAVE] Forecast chart -> {out_png}", fh)

        # ---------- Permutation importance (on residual returns) ----------
        class BoosterWrapper:
            def __init__(self, booster, names):
                self.booster = booster; self.names = names
            def fit(self, X, y=None): return self
            def predict(self, X):
                dm = xgb.DMatrix(X, feature_names=self.names)
                return self.booster.predict(dm)
            def score(self, X, y):
                return r2_score(y, self.predict(X))

        wrapper = BoosterWrapper(booster, feat_names)
        r = permutation_importance(wrapper, X_res_te, y_res_te, n_repeats=20, random_state=RANDOM_STATE)
        imp = pd.Series(r.importances_mean, index=feat_names).sort_values(ascending=False).head(20)
        plt.figure(figsize=(10, 7))
        imp[::-1].plot(kind="barh")
        plt.title("Permutation Importance — residual next-day returns (top 20)")
        plt.tight_layout()
        out_fi = os.path.join(OUT_DIR, "xgb_hybrid_returns_perm_importance.png")
        plt.savefig(out_fi, dpi=150)
        log(f"[SAVE] Perm-importance -> {out_fi}", fh)

        log("Last 5 test rows:", fh)
        log(pd.read_csv(out_csv).tail().to_string(index=False), fh)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n[CRASH] xgb_hybrid_returns.py failed:\n")
        traceback.print_exc()
        sys.exit(1)
