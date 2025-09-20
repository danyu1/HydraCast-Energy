# sarimax_baseline.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

DATA = "data/processed/merged.csv"
OUT_DIR = "outputs"
H = 200  # ---------- 200-day forecast horizon ----------

# Exogenous features to try (only those present will be used)
CAND_EXOG = [
    "crude_inv",
    "yield_spread_10_2",
    "oecd_yoy",
    "payems_yoy",
    "wip_yoy",
    "wti_roll_ann",
    "wti_prompt_spread",
    "ng_roll_ann",
    "ng_prompt_spread",
]

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- Load & prep
    df = pd.read_csv(DATA, index_col="date", parse_dates=True).sort_index()
    if "wti_spot" not in df.columns:
        raise RuntimeError("wti_spot not found in merged.csv")

    y = df["wti_spot"].astype(float).copy()
    exog_cols = [c for c in CAND_EXOG if c in df.columns]
    X = df.reindex(columns=exog_cols).astype(float).ffill()

    aligned = pd.concat([y, X], axis=1).dropna()
    y = aligned["wti_spot"]
    X = aligned.drop(columns=["wti_spot"])

    # --- Build 200d future exogenous (carry-forward last row)
    last_date = y.index[-1]
    future_idx = pd.date_range(last_date + pd.Timedelta(days=1), periods=H, freq="D")
    future_X = None
    if X.shape[1] > 0:
        last_row = X.iloc[-1]
        future_X = pd.DataFrame(
            np.tile(last_row.values, (H, 1)), index=future_idx, columns=X.columns
        )

    # --- Fit SARIMAX
    model = SARIMAX(
        y,
        order=(1, 1, 1),
        seasonal_order=(0, 1, 1, 7),
        exog=X if X.shape[1] > 0 else None,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False)
    print(f"Fitted SARIMAX with exog={list(X.columns)}")
    print(f"AIC={res.aic:.2f}  BIC={res.bic:.2f}  nobs={res.nobs}")

    # --- Forecast 200d
    fc = res.get_forecast(steps=H, exog=future_X)
    pred = fc.predicted_mean
    conf = fc.conf_int()

    # Save CSV
    out_csv = os.path.join(OUT_DIR, "sarimax_200d_forecast.csv")
    pd.DataFrame(
        {"date": future_idx, "forecast": pred.values,
         "lower": conf.iloc[:, 0].values, "upper": conf.iloc[:, 1].values}
    ).to_csv(out_csv, index=False)
    print(f"Saved forecast CSV -> {out_csv}")

    # --- Plots
    # 1) Actual vs Fitted
    plt.figure(figsize=(12, 6))
    plt.plot(y, label="Actual", lw=1.4)
    plt.plot(res.fittedvalues, label="Fitted", alpha=0.7)
    plt.title("Actual vs Fitted Values (SARIMAX)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "sarimax_fitted.png"), dpi=150)

    # 2) Residuals over time
    residuals = res.resid
    plt.figure(figsize=(12, 4))
    plt.plot(residuals, lw=1)
    plt.axhline(0, ls="--", c="k", lw=0.8)
    plt.title("Model Residuals Over Time")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "sarimax_residuals.png"), dpi=150)

    # 3) Residual distribution
    plt.figure(figsize=(6, 4))
    plt.hist(residuals.dropna(), bins=30, edgecolor="black", alpha=0.75)
    plt.title("Residual Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "sarimax_resid_hist.png"), dpi=150)

    # 4) 200d Forecast with fixed y-axis and CI
    plt.figure(figsize=(12, 6))
    plt.plot(y, label="History", lw=1.4)
    plt.plot(pred, label="Forecast (200d)", lw=2)
    plt.fill_between(pred.index, conf.iloc[:, 0], conf.iloc[:, 1], alpha=0.25, label="95% CI")
    plt.ylim(-50, 150)  # ---------- fixed y-axis ----------
    plt.title("200-Day Forecast with Confidence Intervals (SARIMAX)")
    plt.legend()
    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, "sarimax_200d_forecast.png")
    plt.savefig(out_png, dpi=150)

    print(f"Saved charts -> {OUT_DIR}")

if __name__ == "__main__":
    main()
