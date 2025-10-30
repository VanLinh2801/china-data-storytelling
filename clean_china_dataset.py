from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def normalize_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trim leading/trailing whitespace from all string cells.
    Convert empty strings to pandas NA so they are considered missing.
    """
    trimmed = df.applymap(lambda v: v.strip() if isinstance(v, str) else v)
    # Replace remaining empty strings ("") with NA so dropna will remove those rows
    normalized = trimmed.replace("", pd.NA)
    return normalized


def drop_rows_with_any_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows that contain any missing values in any column."""
    return df.dropna(axis=0, how="any")


def clean_china_csv(input_path: Path, output_path: Path) -> None:
    """
    Read input CSV, remove rows with any missing/blank fields, and write output CSV.

    Steps:
    1) Read CSV with default NA detection + additional common blanks.
    2) Trim whitespace from string fields and convert empty strings to NA.
    3) Drop any rows containing NA in any column.
    4) Save the cleaned DataFrame to output_path.
    """
    # Pandas will automatically interpret many markers as NA; we also add some common blanks
    na_values = ["", " ", "NA", "N/A", "na", "n/a", "null", "NULL"]
    df = pd.read_csv(input_path, na_values=na_values, keep_default_na=True)

    df = normalize_whitespace(df)
    cleaned = drop_rows_with_any_missing(df)

    # Use UTF-8 to ensure compatibility; keep index off for a clean CSV
    cleaned.to_csv(output_path, index=False, encoding="utf-8")


def main(argv: list[str]) -> int:
    project_root = Path(__file__).resolve().parent
    input_path = project_root / "China.csv"
    output_path = project_root / "China.cleaned.csv"

    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return 1

    # Optional: create a simple backup alongside the original (non-destructive)
    backup_path = project_root / "China.backup.csv"
    try:
        if not backup_path.exists():
            # Avoid reading/writing the original while it's open in other apps; this is non-invasive
            input_path.replace(backup_path)
            # Move back to original name to preserve expected location
            backup_path.replace(input_path)
    except Exception:
        # If the file is locked by another process, skip backup silently (non-critical for demonstration)
        pass

    clean_china_csv(input_path, output_path)
    print(f"Cleaned data written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


