import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

df = pd.read_csv("data/processed/merged.csv", index_col="date", parse_dates=True)

y = df["wti_spot"]
exog_cols = [
    "crude_inv",
    "yield_spread_10_2",
    "oecd_yoy",
    "payems_yoy",
    "wip_yoy",        #will be absent early; fill handled below
]
X = df.reindex(columns=exog_cols).ffill()

model = SARIMAX(y, order=(1,1,1), seasonal_order=(0,1,1,7), exog=X)
res = model.fit(disp=False)

#true 30-day forward (carry-forward exog except using iterative wti_lag if you add it)
future_X = X.iloc[-1000:].copy()
fc = res.get_forecast(steps=30, exog=future_X)
pred = fc.predicted_mean
conf = fc.conf_int()

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