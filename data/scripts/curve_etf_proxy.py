# curve_eia_xls.py
import os
import io
import requests
import pandas as pd

RAW_DIR = "data/raw"
os.makedirs(RAW_DIR, exist_ok=True)

EIA_XLS = {
    # WTI Crude Oil futures (Cushing, OK)
    "CL1": "https://www.eia.gov/dnav/pet/hist_xls/RCLC1d.xls",
    "CL2": "https://www.eia.gov/dnav/pet/hist_xls/RCLC2d.xls",
    # Henry Hub Natural Gas futures
    "NG1": "https://www.eia.gov/dnav/ng/hist_xls/RNGC1d.xls",
    "NG2": "https://www.eia.gov/dnav/ng/hist_xls/RNGC2d.xls",
}

def read_eia_xls(url: str, label: str) -> pd.Series | None:
    import io, requests, pandas as pd

    try:
        r = requests.get(
            url, timeout=30,
            headers={"User-Agent": "Mozilla/5.0 (HydraCast Energy; +https://example.com)"}
        )
        r.raise_for_status()
    except Exception as e:
        print(f"[warn] {label}: download failed -> {e}")
        return None

    try:
        # Force legacy xls engine
        xls = pd.ExcelFile(io.BytesIO(r.content), engine="xlrd")
        # Prefer 'Data 1' sheet if present; otherwise try all
        sheet_names = ["Data 1"] if "Data 1" in xls.sheet_names else xls.sheet_names

        ser = None
        for sheet in sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet, header=None, engine="xlrd")
            df = df.dropna(how="all").dropna(axis=1, how="all")
            if df.shape[1] < 2:
                continue
            c0 = pd.to_datetime(df.iloc[:, 0], errors="coerce")
            c1 = pd.to_numeric(df.iloc[:, 1], errors="coerce")
            mask = c0.notna() & c1.notna()
            sub = pd.DataFrame({"date": c0, label: c1}).loc[mask]
            if not sub.empty:
                ser = sub.set_index("date")[label].sort_index()
                break

        if ser is None or ser.empty:
            print(f"[warn] {label}: could not parse a date/value block from XLS (sheets tried: {sheet_names})")
            return None

        ser = ser[~ser.index.duplicated(keep="last")]
        ser.index.name = "date"
        return ser
    except Exception as e:
        print(f"[warn] {label}: parse failed -> {e}")
        return None

def calendarize_and_ffill(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_index()
    idx = pd.date_range(df.index.min(), df.index.max(), freq="D")
    return df.reindex(idx).ffill().rename_axis("date")

def main():
    series = {}
    for label, url in EIA_XLS.items():
        s = read_eia_xls(url, label)
        if s is not None:
            series[label] = s

    if not series:
        raise SystemExit("No futures series could be fetched from EIA.")

    df = pd.DataFrame(series)
    df = calendarize_and_ffill(df)

    feats = pd.DataFrame(index=df.index)

    # ----- WTI term structure (needs CL1 & CL2)
    if {"CL1", "CL2"}.issubset(df.columns):
        f1, f2 = df["CL1"], df["CL2"]
        feats["wti_prompt_spread"] = f2 - f1
        feats["wti_slope_pct"] = (f2 / f1) - 1.0
        feats["wti_roll_ann"] = feats["wti_slope_pct"] * 12.0
        feats["wti_contango_flag"] = (feats["wti_prompt_spread"] > 0).astype(int)
        feats["wti_backwardation_flag"] = (feats["wti_prompt_spread"] < 0).astype(int)
    else:
        print("[warn] Missing CL1 or CL2 → skipping WTI curve features.")

    # ----- NatGas term structure (needs NG1 & NG2)
    if {"NG1", "NG2"}.issubset(df.columns):
        g1, g2 = df["NG1"], df["NG2"]
        feats["ng_prompt_spread"] = g2 - g1
        feats["ng_slope_pct"] = (g2 / g1) - 1.0
        feats["ng_roll_ann"] = feats["ng_slope_pct"] * 12.0
        feats["ng_contango_flag"] = (feats["ng_prompt_spread"] > 0).astype(int)
        feats["ng_backwardation_flag"] = (feats["ng_prompt_spread"] < 0).astype(int)
    else:
        print("[warn] Missing NG1 or NG2 → skipping NatGas curve features.")

    if feats.empty:
        raise SystemExit("No curve features computed (missing both WTI and NG legs).")

    out_path = os.path.join(RAW_DIR, "curve.csv")
    feats.to_csv(out_path)
    print(f"Saved futures term-structure features → {out_path}")
    print(feats.tail())

if __name__ == "__main__":
    main()
