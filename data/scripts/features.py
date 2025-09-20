import pandas as pd

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    #Forward-fill inventories (weekly → daily)
    df["crude_inv"] = df["crude_inv"].ffill()

    #Lag features by 1 day and 7 days
    df["wti_lag1"] = df["wti_spot"].shift(1)
    df["inv_lag7"] = df["crude_inv"].shift(7)

    #Calendar features
    df["dow"] = df.index.dayofweek   #0=Mon, 6=Sun
    df["month"] = df.index.month

    return df.dropna()

df = pd.read_csv("data/raw/wti_inv_sample.csv", index_col="date", parse_dates=True)
df = build_features(df)
print(df)