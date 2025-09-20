# src/data/eia_test.py
import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
KEY = os.getenv("EIA_API_KEY")
assert KEY, "Missing EIA_API_KEY in .env"

def eia_series(series_id: str, alias: str) -> pd.DataFrame:
    """Fetch a single EIA v2 series and return a 1-column dataframe named `alias`."""
    url = f"https://api.eia.gov/v2/seriesid/{series_id}?api_key={KEY}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    j = r.json()
    data = j["response"]["data"]
    # Keep ONLY period + value to avoid overlapping columns on join
    df = pd.DataFrame(data)[["period", "value"]]
    df["period"] = pd.to_datetime(df["period"])
    df = df.rename(columns={"period": "date", "value": alias})
    return df.set_index("date").sort_index()

def main():
    # WTI spot (daily) + weekly inventories
    wti = eia_series("PET.RWTC.D", "wti_spot")
    inv = eia_series("PET.WCESTUS1.W", "crude_inv")

    # Make both daily; WTI daily already, but enforce and forward-fill just in case
    wti = wti.asfreq("D").ffill()

    # Inventories are weekly; resample to daily and forward-fill (replaces deprecated interpolate('pad'))
    inv = inv.resample("D").ffill()

    out = wti.join(inv, how="left")
    out.to_csv("data/raw/wti_inv_sample.csv")
    print("Saved -> data/raw/wti_inv_sample.csv")
    print(out.tail())

if __name__ == "__main__":
    main()
