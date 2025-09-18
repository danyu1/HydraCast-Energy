# curve_etf_proxy.py  (run from project root)
import os, time
import pandas as pd
import yfinance as yf

RAW_DIR = "data/raw"

def _download_one(ticker: str, start="2008-01-01", max_retries=3, sleep_sec=2) -> pd.Series | None:
    """
    Download a single ticker with retries.
    Uses auto_adjust=False so 'Adj Close' is present; falls back to 'Close'.
    Returns a Series indexed by date named as the ticker, or None if failed.
    """
    for i in range(max_retries):
        try:
            df = yf.download(
                ticker, start=start,
                auto_adjust=False, progress=False, threads=False, group_by="column"
            )
            if df is None or df.empty:
                raise RuntimeError("empty dataframe")

            # Handle both single and MultiIndex column cases
            s = None
            if isinstance(df.columns, pd.MultiIndex):
                # Try to get the level named 'Adj Close' or 'Close'
                for colname in ("Adj Close", "Close"):
                    try:
                        s = df.xs(colname, axis=1, level=-1)
                        break
                    except Exception:
                        pass
                if s is None:
                    # flatten and try again
                    flat = {"_".join(map(str, c)).strip(): c for c in df.columns}
                    for colname in ("Adj Close", "Close"):
                        for flat_name, tpl in flat.items():
                            if flat_name.endswith(colname):
                                s = df.loc[:, tpl]
                                break
                        if s is not None:
                            break
            else:
                # Single-level columns
                if "Adj Close" in df.columns:
                    s = df["Adj Close"].copy()
                elif "Close" in df.columns:
                    s = df["Close"].copy()

            if s is None or s.empty:
                raise KeyError("no usable Close/Adj Close column found")

            s.name = ticker
            s.index.name = "date"
            return s
        except Exception as e:
            if i == max_retries - 1:
                print(f"[warn] {ticker}: failed after {max_retries} tries -> {e}")
                return None
            time.sleep(sleep_sec)

def fetch_curve_proxy(start="2008-01-01") -> pd.DataFrame:
    """
    ETF-based proxies for term structure:
      - USO ~ front-month crude
      - USL ~ 12-month ladder crude
      - UNG ~ front-month nat gas
    Features:
      - wti_etf_ratio = USO/USL
      - wti_etf_ratio_z = z-score of ratio (252d)
      - ung_ret_21d = 21d return of UNG
    """
    series = {}
    for tkr in ("USO", "USL", "UNG"):
        s = _download_one(tkr, start=start)
        if s is not None:
            series[tkr] = s

    if not series:
        raise RuntimeError("No ETF data available (USO/USL/UNG all failed).")

    df = pd.DataFrame(series)
    # daily index; forward-fill weekends/holidays so it aligns with your daily pipeline
    df = df.asfreq("D").ffill()

    feats = pd.DataFrame(index=df.index)

    # WTI term-structure proxy (needs both USO & USL)
    if {"USO", "USL"}.issubset(df.columns):
        feats["wti_etf_ratio"] = df["USO"] / df["USL"]
        rollwin = 252
        feats["wti_etf_ratio_z"] = (
            (feats["wti_etf_ratio"] - feats["wti_etf_ratio"].rolling(rollwin).mean())
            / feats["wti_etf_ratio"].rolling(rollwin).std()
        )

    # NG proxy (UNG)
    if "UNG" in df.columns:
        feats["ung_ret_21d"] = df["UNG"].pct_change(21)

    if feats.empty:
        raise RuntimeError("Could not compute any features (missing USO/USL and UNG).")

    feats.index.name = "date"
    return feats

def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    feats = fetch_curve_proxy()
    out_path = os.path.join(RAW_DIR, "curve_proxy.csv")
    feats.to_csv(out_path)
    print(f"Saved curve proxy features -> {out_path}")
    print(feats.tail())

if __name__ == "__main__":
    main()
