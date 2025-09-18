import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from features import build_features

#Load raw CSV
df = pd.read_csv("data/raw/wti_inv_sample.csv", index_col="date", parse_dates=True)
df = build_features(df)


#Target & exogenous features
y = df["wti_spot"]
X = df[["crude_inv", "wti_lag1", "inv_lag7", "dow", "month"]]

#Fit SARIMAX (ARIMA(1,1,1) with weekly seasonality)
model = SARIMAX(
    y,
    order=(1,1,1),
    seasonal_order=(0,1,1,7),
    exog=X
)
res = model.fit(disp=False)

#Forecast 30 days ahead
future_X = X.iloc[-7265:].copy()
forecast = res.get_forecast(steps=7265, exog=future_X)
pred = forecast.predicted_mean
conf = forecast.conf_int()

print("30-day forecast:")
print(pd.DataFrame({"forecast": pred, "lower": conf.iloc[:,0], "upper": conf.iloc[:,1]}))

import matplotlib.pyplot as plt
# 1.Actual vs. Fitted values
plt.figure(figsize=(12,6))
plt.plot(y, label="Actual", color="blue")
plt.plot(res.fittedvalues, label="Fitted", color="red", alpha=0.7)
plt.title("Actual vs Fitted Values (SARIMAX)")
plt.legend()
plt.show()

# 2.Residuals over time
residuals = res.resid
plt.figure(figsize=(12,4))
plt.plot(residuals, color="purple")
plt.axhline(0, linestyle="--", color="black")
plt.title("Model Residuals Over Time")
plt.show()

# 3.Residual distribution
plt.figure(figsize=(6,4))
plt.hist(residuals, bins=30, color="grey", edgecolor="black", alpha=0.7)
plt.title("Residual Distribution")
plt.show()

# 4.Forecast with confidence intervals
plt.figure(figsize=(12,6))
plt.plot(y, label="History", color="blue")
plt.plot(pred, label="Forecast", color="red")
plt.fill_between(pred.index, conf.iloc[:,0], conf.iloc[:,1], 
                 color="pink", alpha=0.3, label="95% CI")
plt.title("30-Day Forecast with Confidence Intervals")
plt.ylim(-50, 150)   # fix y-axis scale
plt.legend()
plt.show()