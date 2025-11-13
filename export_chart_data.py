#!/usr/bin/env python3
"""
Generate the CSV tables that act as sources for each visualization in analyze_temperature.py.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_FILE = Path(__file__).with_name("data.csv")
OUT_DIR = Path("derived_data")


def load_temperature_data(path: Path) -> pd.DataFrame:
    """Load and tidy the raw JMA CSV."""
    df = pd.read_csv(
        path,
        encoding="shift_jis",
        skiprows=5,
        names=["date", "avg_temp_c", "quality", "homogenization_flag"],
    )
    df["date"] = pd.to_datetime(df["date"], format="%Y/%m/%d")
    df = (
        df.dropna(subset=["avg_temp_c"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    return df


def export_daily_series(df: pd.DataFrame, outdir: Path) -> None:
    """Export the cleaned daily dataset plus the global mean."""
    mean_temp = df["avg_temp_c"].mean()
    daily = df[["date", "avg_temp_c", "quality", "homogenization_flag"]].copy()
    daily["overall_mean_temp"] = mean_temp
    daily["date"] = daily["date"].dt.strftime("%Y-%m-%d")
    daily.to_csv(outdir / "daily_series.csv", index=False)


def export_monthly_means(df: pd.DataFrame, outdir: Path) -> None:
    """Export monthly mean temperatures (month-end timestamps)."""
    monthly = df.set_index("date")["avg_temp_c"].resample("ME").mean().reset_index()
    monthly.rename(columns={"date": "month_end", "avg_temp_c": "mean_temp_c"}, inplace=True)
    monthly["month_end"] = monthly["month_end"].dt.strftime("%Y-%m-%d")
    monthly.to_csv(outdir / "monthly_means.csv", index=False)


def export_rolling_trend(df: pd.DataFrame, outdir: Path, window: int = 30) -> None:
    """Export daily values along with the rolling mean used for the trend plot."""
    rolling_df = df[["date", "avg_temp_c"]].copy()
    rolling_df.sort_values("date", inplace=True)
    rolling_df["rolling_mean_30d"] = (
        rolling_df["avg_temp_c"].rolling(window=window, min_periods=window // 2).mean()
    )
    rolling_df["date"] = rolling_df["date"].dt.strftime("%Y-%m-%d")
    rolling_df.to_csv(outdir / "rolling_trend.csv", index=False)


def export_monthly_box_stats(df: pd.DataFrame, outdir: Path) -> None:
    """Summaries used by the monthly box plot (quartiles per calendar month)."""
    quantiles = df.groupby("month")["avg_temp_c"].quantile([0.25, 0.5, 0.75]).unstack()
    quantiles.columns = ["q1", "median", "q3"]
    summary = df.groupby("month")["avg_temp_c"].agg(["count", "min", "max", "mean"])
    stats = summary.join(quantiles).reset_index().sort_values("month")
    stats.rename(columns={"mean": "mean_temp"}, inplace=True)
    stats.to_csv(outdir / "monthly_boxplot_stats.csv", index=False)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    df = load_temperature_data(DATA_FILE)
    export_daily_series(df, OUT_DIR)
    export_monthly_means(df, OUT_DIR)
    export_rolling_trend(df, OUT_DIR)
    export_monthly_box_stats(df, OUT_DIR)
    print(f"Wrote CSV files to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
