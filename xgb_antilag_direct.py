# xgb_antilag_direct.py
# Anti-lag direct next-day forecaster with robust NaN handling & auto-pruning of bad columns.

import os, sys, traceback
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime

DATA = "data/processed/merged.csv"
OUT_DIR = "outputs"

# Splits by TARGET DAY (what we forecast)
H_TEST = 30
H_VAL  = 90
RANDOM_STATE = 32

# Column must have at least this many non-null rows to be used
MIN_NON_NULL = 200   # adjust if your dataset is very small

def log(msg):
    print(msg, flush=True)

def dir_acc(y_true, y_pred):
    dyt = np.sign(np.diff(y_true))
    dyp = np.sign(np.diff(y_pred))
    return float((dyt == dyp).mean()) if len(dyt) else float("nan")

def build_fast_features(df: pd.DataFrame) -> pd.DataFrame:
    """Only fast/reactive features available at time t."""
    f = df.copy()

    # Short lags of the level (kept small)
    for l in (1, 2, 3, 5, 7):
        f[f"wti_lag{l}"] = f["wti_spot"].shift(l)

    # Very short returns / momentum (differences, not pct)
    f["ret_1"] = f["wti_spot"].diff(1)
    f["ret_2"] = f["wti_spot"].diff(2)
    f["ret_3"] = f["wti_spot"].diff(3)
    f["ret_5"] = f["wti_spot"].diff(5)
    f["ret_7"] = f["wti_spot"].diff(7)

    # “Fast” macro/curve — raw and tiny lags only (include only if present)
    macro_raw = [
        "crude_inv", "wti_prompt_spread", "wti_roll_ann",
        "ng_prompt_spread", "ng_roll_ann",
        "yield_spread_10_2", "oecd_yoy", "payems_yoy", "wip_yoy",
    ]
    macro_raw = [c for c in macro_raw if c in f.columns]
    for c in macro_raw:
        f[f"{c}_lag1"] = f[c].shift(1)
        f[f"{c}_lag3"] = f[c].shift(3)

    # Day-of-week & month
    f["dow"] = f.index.dayofweek
    f["month"] = f.index.month

    # Residual lags (placeholders for now; we fill after AR step)
    f["res_lag1"] = np.nan
    f["res_lag2"] = np.nan
    f["res_lag3"] = np.nan

    return f

def main():
    try:
        os.makedirs(OUT_DIR, exist_ok=True)
        log(f"\n===== Anti-lag run @ {datetime.now()} =====")

        if not os.path.exists(DATA):
            log(f"[FATAL] Missing {DATA}"); sys.exit(1)

        df = pd.read_csv(DATA, index_col="date", parse_dates=True).sort_index().ffill()
        if "wti_spot" not in df.columns:
            log("[FATAL] 'wti_spot' missing in merged.csv"); sys.exit(1)

        # ---------------- Build base features ----------------
        f = build_fast_features(df)

        # Quick AR(1..3) to derive residuals (reactive baseline)
        ar_cols_small = [c for c in ["wti_lag1","wti_lag2","wti_lag3"] if c in f.columns]
        ar_frame = f[["wti_spot"] + ar_cols_small].dropna()
        if ar_frame.empty:
            log("[WARN] AR frame is empty (no wti_lag1..3?). Skipping residual features.")
        ar_hat_full = pd.Series(np.nan, index=f.index, dtype=float)
        if not ar_frame.empty:
            ar = LinearRegression()
            ar.fit(ar_frame[ar_cols_small], ar_frame["wti_spot"])
            ar_hat_full.loc[ar_frame.index] = ar.predict(ar_frame[ar_cols_small])

            # Residual lags now that AR is known
            res = (df["wti_spot"] - ar_hat_full)
            f.loc[:, "res_lag1"] = res.shift(1)
            f.loc[:, "res_lag2"] = res.shift(2)
            f.loc[:, "res_lag3"] = res.shift(3)

        # ---------------- Auto-prune unusable columns ----------------
        # Count non-nulls per feature col; keep only those with enough data
        feat_cols_all = [c for c in f.columns if c != "wti_spot"]
        nn = f[feat_cols_all].notna().sum().sort_values()
        bad = nn[nn < MIN_NON_NULL].index.tolist()
        good = [c for c in feat_cols_all if c not in bad]

        log("[INFO] Non-null counts (tail shows weakest columns):")
        preview = nn.to_frame("non_null").tail(20)
        log(preview.to_string())

        if bad:
            log(f"[INFO] Dropping {len(bad)} sparse columns (min_non_null={MIN_NON_NULL}): {bad[:15]}{' ...' if len(bad)>15 else ''}")

        f_use = pd.concat([df[["wti_spot"]], f[good]], axis=1)

        # ---------------- Align target (t -> y[t+1]) ----------------
        y_next = df["wti_spot"].shift(-1).rename("y_next")
        design = pd.concat([y_next, f_use.drop(columns=["wti_spot"])], axis=1).dropna()

        if design.empty:
            # Try again without residual lags if they were too sparse
            no_res_cols = [c for c in f_use.columns if not c.startswith("res_lag")]
            design = pd.concat([y_next, f_use[no_res_cols].drop(columns=["wti_spot"], errors="ignore")], axis=1).dropna()

        if design.empty:
            log(f"[FATAL] Not enough rows after alignment (0). Try lowering MIN_NON_NULL or check NaNs in your macro columns.")
            sys.exit(1)

        idx_all_targ = design.index
        if len(idx_all_targ) <= (H_VAL + H_TEST + 50):
            log(f"[FATAL] Not enough rows after alignment ({len(idx_all_targ)}).")
            sys.exit(1)

        idx_train_targ = idx_all_targ[:-(H_VAL + H_TEST)]
        idx_val_targ   = idx_all_targ[-(H_VAL + H_TEST):-H_TEST]
        idx_test_targ  = idx_all_targ[-H_TEST:]

        feat_names = [c for c in design.columns if c != "y_next"]
        X_tr = design.loc[idx_train_targ, feat_names].astype(float).values
        y_tr = design.loc[idx_train_targ, "y_next"].astype(float).values

        X_val = design.loc[idx_val_targ, feat_names].astype(float).values
        y_val = design.loc[idx_val_targ, "y_next"].astype(float).values

        X_te = design.loc[idx_test_targ, feat_names].astype(float).values
        y_te = design.loc[idx_test_targ, "y_next"].astype(float).values

        log(f"[INFO] Using {len(feat_names)} features. Splits (target days) -> train={len(y_tr)}, val={len(y_val)}, test={len(y_te)}")

        # ---------------- XGBoost (direct next-day) ----------------
        dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=feat_names)
        dval   = xgb.DMatrix(X_val, label=y_val, feature_names=feat_names)
        dtest  = xgb.DMatrix(X_te, feature_names=feat_names)

        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "eta": 0.05,
            "max_depth": 6,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.4,
            "reg_lambda": 1.2,
            "gamma": 0.3,
            "tree_method": "hist",
            "seed": RANDOM_STATE,
        }

        booster = xgb.train(
            params, dtrain,
            num_boost_round=4000,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=200,
            verbose_eval=False,
        )
        log(f"[INFO] Best iteration: {booster.best_iteration + 1}")

        val_pred_raw = booster.predict(dval)
        te_pred_raw  = booster.predict(dtest)

        # ---------------- Anti-lag debiaser (from validation) ----------------
        def add_pred_features(pred):
            dp = np.r_[np.nan, np.diff(pred)]
            return np.column_stack([pred, np.nan_to_num(dp, nan=0.0)])

        Z_val = add_pred_features(val_pred_raw)
        Z_te  = add_pred_features(te_pred_raw)

        debias = LinearRegression()
        debias.fit(Z_val, y_val)
        val_pred = debias.predict(Z_val)
        te_pred  = debias.predict(Z_te)

        # ---------------- Metrics ----------------
        def report(split, y_true, y_hat):
            rmse = float(np.sqrt(mean_squared_error(y_true, y_hat)))
            mae  = float(mean_absolute_error(y_true, y_hat))
            r2   = float(r2_score(y_true, y_hat))
            dacc = dir_acc(y_true, y_hat)
            log(f"[{split}] RMSE={rmse:.3f}  MAE={mae:.3f}  R2={r2:.3f}  DirAcc={dacc:.0%}")
            return rmse, mae, r2, dacc

        log("[VAL] Raw (pre-debias):")
        report("VAL-RAW", y_val, val_pred_raw)
        log("[VAL] Debiased:")
        report("VAL    ", y_val, val_pred)

        log("[TEST] Raw (pre-debias):")
        report("TEST-RAW", y_te, te_pred_raw)
        log("[TEST] Debiased:")
        rmse, mae, r2, dacc = report("TEST    ", y_te, te_pred)

        # ---------------- Save ----------------
        os.makedirs(OUT_DIR, exist_ok=True)
        out_csv = os.path.join(OUT_DIR, "xgb_antilag_direct_30d.csv")
        pd.DataFrame({
            "date": idx_test_targ,
            "forecast_raw": te_pred_raw,
            "forecast": te_pred,
            "actual": y_te,
        }).to_csv(out_csv, index=False)
        log(f"[SAVE] Forecast CSV -> {out_csv}")

        plt.figure(figsize=(12, 6))
        plt.plot(idx_test_targ, y_te, label="Actual", lw=1.4)
        plt.plot(idx_test_targ, te_pred, label="Forecast (debias)", lw=2)
        plt.plot(idx_test_targ, te_pred_raw, label="Forecast (raw)", lw=1, alpha=0.6)
        plt.title(f"Anti-lag direct 30d | RMSE={rmse:.2f}  MAE={mae:.2f}  DirAcc={dacc:.0%}")
        plt.legend(); plt.tight_layout()
        out_png = os.path.join(OUT_DIR, "xgb_antilag_direct_30d.png")
        plt.savefig(out_png, dpi=150)
        log(f"[SAVE] Chart -> {out_png}")

        # Feature gains
        try:
            score = booster.get_score(importance_type="gain")
            if score:
                fi = (
                    pd.Series(score)
                    .sort_values(ascending=False)
                    .rename_axis("feature")
                    .reset_index(name="gain")
                )
                fi_path = os.path.join(OUT_DIR, "xgb_antilag_direct_feature_gain.csv")
                fi.to_csv(fi_path, index=False)
                log(f"[SAVE] Feature gains -> {fi_path}")
        except Exception:
            pass

        log("Done.")

    except Exception:
        print("\n[CRASH] xgb_antilag_direct.py failed:\n")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
