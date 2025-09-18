# src/experiments/sarimax_grid.py
import warnings, itertools, numpy as np, pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

warnings.filterwarnings("ignore")

DATA_PATH = "data/processed/merged.csv"   # created by merge_data.py

# ------------------------------
# Utilities
# ------------------------------
def rmse(a,b): return float(np.sqrt(mean_squared_error(a,b)))
def mae(a,b):  return float(mean_absolute_error(a,b))
def mape(a,b): return float(mean_absolute_percentage_error(a,b))

def try_fit(y, X, order=(1,1,1), sorder=(0,1,1,7)):
    """Fit SARIMAX safely; return fitted model or None."""
    try:
        model = SARIMAX(
            y, exog=X,
            order=order, seasonal_order=sorder,
            enforce_stationarity=False, enforce_invertibility=False
        )
        res = model.fit(disp=False)
        return res
    except Exception as e:
        print("Fit failed:", e)
        return None

def safe_walk_forward(df, target, exog_cols, horizon=30, stride=5):
    """
    Builds a per-feature-set subset, picks a safe horizon/start,
    runs expanding-window CV, returns metrics dict or None.
    """
    # keep only required cols and drop NaNs just for these
    cols = [target] + exog_cols
    sub = df.loc[:, [c for c in cols if c in df.columns]].dropna()
    if len(sub) < 40:  # not enough data for any meaningful CV
        return None

    # dynamic horizon (<= 30), at most 10% of data, at least 7
    H = int(min(horizon, max(7, len(sub) // 10)))
    # training start after ~60% of data (but with room for 1 horizon)
    start_test_idx = max(int(len(sub) * 0.6), 30)
    start_test_idx = min(start_test_idx, len(sub) - (H + 1))
    if start_test_idx < 1:
        return None

    y = sub[target]
    X = sub[exog_cols] if exog_cols else None

    metrics, aics, runs, ok = [], [], 0, 0
    t = start_test_idx
    while t + H <= len(sub):
        runs += 1
        y_tr, y_te = y.iloc[:t], y.iloc[t:t+H]
        X_tr = X.iloc[:t] if X is not None else None
        X_te = X.iloc[t:t+H] if X is not None else None
        try:
            res = SARIMAX(
                y_tr, exog=X_tr,
                order=(1,1,1), seasonal_order=(0,1,1,7),
                enforce_stationarity=False, enforce_invertibility=False
            ).fit(disp=False)
            pred = res.get_forecast(steps=len(y_te), exog=X_te).predicted_mean
            metrics.append((rmse(y_te, pred), mae(y_te, pred), mape(y_te, pred)))
            aics.append(float(res.aic)); ok += 1
        except Exception as e:
            print(f"[skip] exog={exog_cols or 'None'} t={t}: {e}")
        t += stride

    if not metrics: return None
    R, M, P = zip(*metrics)
    return {
        "rows": len(sub), "horizon": H, "runs": runs, "ok": ok,
        "rmse": float(np.mean(R)), "mae": float(np.mean(M)),
        "mape": float(np.mean(P)), "aic_mean": float(np.mean(aics)),
        "exog": exog_cols
    }

def main():
    df = pd.read_csv(DATA_PATH, index_col="date", parse_dates=True).sort_index()
    df = df.ffill()  # macro is slower frequency; ffill is fine

    # optional macro lags (created before selection)
    for c in ["oecd_yoy","oecd_z","payems_yoy","wip_yoy"]:
        if c in df.columns:
            df[c+"_lag30"] = df[c].shift(30)

    target = "wti_spot"

    CANDS = [
        [],  # ARIMA-only
        ["crude_inv"],
        ["crude_inv", "yield_spread_10_2"],
        ["crude_inv", "yield_spread_z"],
        ["crude_inv", "oecd_yoy_lag30"],
        ["crude_inv", "payems_yoy_lag30"],
        ["crude_inv", "oecd_yoy_lag30", "yield_spread_z"],
        ["crude_inv", "oecd_yoy_lag30", "payems_yoy_lag30", "yield_spread_z"],
    ]
    # keep only existing columns
    CANDS = [[c for c in cols if c in df.columns] for cols in CANDS]

    results = []
    for exog in CANDS:
        r = safe_walk_forward(df, target, exog, horizon=30, stride=5)
        if r: results.append(r)

    if not results:
        print("No successful fits (data subset too small). Quick checks:")
        print("len(df) =", len(df), " first=", df.index.min(), " last=", df.index.max())
        print("NA counts:\n", df.isna().sum().sort_values(ascending=False).head(15))
        return

    out = pd.DataFrame(results).sort_values("rmse")
    print("\n=== HydraCast SARIMAX Leaderboard (lower is better) ===")
    print(out[["rows","horizon","runs","ok","rmse","mae","mape","aic_mean","exog"]].to_string(index=False))

if __name__ == "__main__":
    main()
