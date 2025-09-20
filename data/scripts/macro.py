import io
import requests
import pandas as pd
from pandas_datareader import data as pdr

# ---------- Helpers ----------
def fetch_fred(series_code: str, start="2000-01-01") -> pd.DataFrame:
    df = pdr.DataReader(series_code, "fred", start)
    df = df.rename(columns={series_code: series_code})
    df = df[~df.index.duplicated(keep="last")]          # <- de-dupe here
    return df

def fetch_oecd_cli(country="OECD") -> pd.DataFrame:
    url = f"https://stats.oecd.org/sdmx-json/data/MEI_CLI/LOLITOAA.{country}.M/all?contentType=csv"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    raw = pd.read_csv(io.StringIO(r.text))

    colmap = {c.lower(): c for c in raw.columns}
    time_col = colmap.get("time_period") or colmap.get("time") or colmap.get("period")
    val_col  = colmap.get("value") or colmap.get("obs_value")
    freq_col = colmap.get("freq") or colmap.get("frequency")
    if not time_col or not val_col:
        raise KeyError(f"Unexpected OECD columns: {list(raw.columns)}")

    if freq_col in raw.columns:
        df = raw[raw[freq_col].astype(str).str.upper().str.startswith("M")].copy()
    else:
        df = raw[~raw[time_col].astype(str).str.contains("Q", na=False)].copy()

    df = df[[time_col, val_col]].rename(columns={time_col: "date", val_col: "OECDLEAD"})
    df["date"] = pd.to_datetime(df["date"].astype(str), format="mixed", errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="last")]          # <- de-dupe here
    return df

def load_macro(start="2000-01-01") -> pd.DataFrame:
    # ----- Fetch base series -----
    fred_codes = ["DGS2", "DGS10", "PAYEMS"]
    fred_dfs = [fetch_fred(code, start) for code in fred_codes]
    macro = pd.concat(fred_dfs, axis=1)

    # World Industrial Production (quarterly), optional
    try:
        wip = fetch_fred("IPB50001SQ", start)  # quarterly
        macro = macro.join(wip, how="outer")
    except Exception:
        pass

    # OECD CLI (monthly)
    oecd = fetch_oecd_cli("OECD")              # monthly
    macro = macro.join(oecd, how="outer")

    # Clean up, numeric, dedupe, sorted
    macro = macro.apply(pd.to_numeric, errors="coerce")
    macro.index = pd.to_datetime(macro.index)
    macro = macro.sort_index()
    macro = macro.groupby(level=0).last()

    # ====== BUILD FEATURES ON NATIVE FREQUENCIES ======

    # 10y-2y spread (daily → daily)
    dgs2 = macro["DGS2"]
    dgs10 = macro["DGS10"]
    spread = dgs10 - dgs2
    spread_z = (spread - spread.rolling(252).mean()) / spread.rolling(252).std()

    # PAYEMS (monthly): compute MoM/YoY on monthly closes, then daily-ffill
    payems_m = macro["PAYEMS"].resample("M").last()
    payems_mom = payems_m.pct_change(fill_method=None)
    payems_yoy = payems_m.pct_change(12, fill_method=None)

    # OECD CLI (monthly): YoY & z on monthly, then daily-ffill
    oecd_m = macro["OECDLEAD"].resample("M").last()
    oecd_yoy = oecd_m.pct_change(12, fill_method=None)
    oecd_z = (oecd_m - oecd_m.rolling(12).mean()) / oecd_m.rolling(12).std()

    # WIP (quarterly): YoY on quarterly, then daily-ffill (optional)
    if "IPB50001SQ" in macro.columns:
        wip_q = macro["IPB50001SQ"].resample("Q").last()
        wip_yoy_q = wip_q.pct_change(4, fill_method=None)  # YoY (4 quarters)
    else:
        wip_q = None
        wip_yoy_q = None

    # ====== DAILY ALIGNMENT ======
    # Build a daily index spanning all series
    start_day = macro.index.min()
    end_day = macro.index.max()
    daily_idx = pd.date_range(start_day, end_day, freq="D")

    out = pd.DataFrame(index=daily_idx)
    out["DGS2"] = dgs2.reindex(daily_idx).ffill()
    out["DGS10"] = dgs10.reindex(daily_idx).ffill()
    out["yield_spread_10_2"] = spread.reindex(daily_idx).ffill()
    out["yield_spread_z"] = spread_z.reindex(daily_idx).ffill()

    # Map monthly/quarterly transforms to daily by forward-filling
    out["PAYEMS"] = payems_m.reindex(daily_idx).ffill()
    out["payems_mom"] = payems_mom.reindex(daily_idx).ffill()
    out["payems_yoy"] = payems_yoy.reindex(daily_idx).ffill()

    out["OECDLEAD"] = oecd_m.reindex(daily_idx).ffill()
    out["oecd_yoy"] = oecd_yoy.reindex(daily_idx).ffill()
    out["oecd_z"] = oecd_z.reindex(daily_idx).ffill()

    if wip_q is not None:
        out["IPB50001SQ"] = wip_q.reindex(daily_idx).ffill()
        out["wip_yoy"] = wip_yoy_q.reindex(daily_idx).ffill()

    # Drop rows that are entirely NaN at the very start
    out = out.dropna(how="all")

    # Quick sanity: ensure all numeric
    out = out.apply(pd.to_numeric, errors="coerce")

    return out


if __name__ == "__main__":
    import os
    os.makedirs("data/raw", exist_ok=True)
    df = load_macro()
    df.to_csv("data/raw/macro.csv")
    print("Saved macro indicators -> data/raw/macro.csv")
    print(df.tail())
