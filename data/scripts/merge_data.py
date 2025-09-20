# merge_data.py  (run from project root)
import os
import pandas as pd

RAW_DIR = "data/raw"
OUT_DIR = "data/processed"
OUT_PATH = os.path.join(OUT_DIR, "merged.csv")

os.makedirs(OUT_DIR, exist_ok=True)

def read_any_csv(path: str, index_name: str = "date") -> pd.DataFrame:
    try:
        df = pd.read_csv(path, index_col=index_name, parse_dates=True)
    except Exception:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df

def describe(name: str, df: pd.DataFrame):
    rng = (str(df.index.min().date()), str(df.index.max().date())) if len(df) else ("-", "-")
    print(f"[{name}] rows={len(df)}  range={rng}")
    print(f"  cols: {list(df.columns)[:8]}{' ...' if len(df.columns)>8 else ''}")

def main():
    # --- Required
    wti_inv = read_any_csv(os.path.join(RAW_DIR, "wti_inv_sample.csv"))  # expects 'wti_spot','crude_inv'
    macro   = read_any_csv(os.path.join(RAW_DIR, "macro.csv"))

    # --- Optional curve
    curve = None
    if os.path.exists(os.path.join(RAW_DIR, "curve.csv")):
        curve = read_any_csv(os.path.join(RAW_DIR, "curve.csv"))
        curve_source = "curve.csv"
    elif os.path.exists(os.path.join(RAW_DIR, "curve_proxy.csv")):
        curve = read_any_csv(os.path.join(RAW_DIR, "curve_proxy.csv"))
        curve_source = "curve_proxy.csv"
    else:
        curve_source = None

    # --- Show diagnostics BEFORE merging
    describe("WTI+INV", wti_inv)
    describe("MACRO", macro)
    if curve is not None:
        describe("CURVE", curve)
        # show a few curve columns so you know names
        print("  sample CURVE cols:", [c for c in curve.columns][:10])
    else:
        print("[CURVE] not found")

    # --- Merge (LEFT join on WTI to keep target & preserve curve columns)
    merged = wti_inv.join(macro, how="left")
    if curve is not None:
        merged = merged.join(curve, how="left")

    # forward-fill slower features (macro/curve)
    merged = merged.ffill()

    # keep rows where target exists
    if "wti_spot" not in merged.columns:
        raise KeyError("Expected 'wti_spot' in merged dataset but it was not found.")
    merged = merged.dropna(subset=["wti_spot"])

    # save
    merged.to_csv(OUT_PATH)
    print(f"\nSaved -> {OUT_PATH}")

    # --- Post-merge checks
    describe("MERGED", merged)

    if curve is not None:
        curve_cols = list(curve.columns)
        present = [c for c in curve_cols if c in merged.columns]
        missing = [c for c in curve_cols if c not in merged.columns]
        print("\nCurve columns present in MERGED:", present[:12], "..." if len(present)>12 else "")
        if missing:
            print("Curve columns missing (unexpected):", missing)

        # Show last non-NA date for a couple curve fields (helps confirm coverage)
        probe = [c for c in curve_cols if "wti_" in c or "ng_" in c][:5]
        if probe:
            print("\nLast non-NA date for sample curve cols:")
            for c in probe:
                print(f"  {c:24s} -> {merged[c].last_valid_index()}")

if __name__ == "__main__":
        main()
