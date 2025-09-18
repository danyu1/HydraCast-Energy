# src/data/merge_data.py
import os
import pandas as pd

RAW_DIR = "data/raw"
os.makedirs("data/processed", exist_ok=True)

def read_wti_inv(path):
    # This one *does* have 'date' as a column name from your earlier save
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    df.index.name = "date"
    return df

def read_macro(path):
    """
    Read macro.csv regardless of whether the index column is named 'date'
    or saved as an unnamed first column.
    """
    try:
        df = pd.read_csv(path, index_col="date", parse_dates=True)
    except ValueError:
        # Fallback: use first column as index
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    # Ensure datetime index + name
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df

def main():
    wti_inv = read_wti_inv(f"{RAW_DIR}/wti_inv_sample.csv")
    macro   = read_macro(f"{RAW_DIR}/macro.csv")

    # Align to overlapping dates and forward-fill small macro gaps
    merged = wti_inv.join(macro, how="inner").sort_index()
    merged = merged.ffill()  # safe: macro series are slower freq; WTI/inv already daily
    # Keep only rows where target exists
    merged = merged.dropna(subset=["wti_spot"])

    out_path = "data/processed/merged.csv"
    merged.to_csv(out_path)
    print(f"Saved -> {out_path}")
    print(merged.tail())

if __name__ == "__main__":
    main()
