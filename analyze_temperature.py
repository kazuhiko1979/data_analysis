#!/usr/bin/env python3
"""
Analyze Tokyo mean temperature observations and generate multiple figures.
"""
from __future__ import annotations

import os
from pathlib import Path

MPL_CACHE_DIR = Path(__file__).with_name(".matplotlib_cache")
MPL_CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib.pyplot as plt
import pandas as pd

DATA_FILE = Path(__file__).with_name("data.csv")
FIG_DIR = Path("figures")


def load_temperature_data(path: Path) -> pd.DataFrame:
    """Load the CSV provided by JMA and return a cleaned dataframe."""
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
    df["dayofyear"] = df["date"].dt.dayofyear
    return df


def summarize(df: pd.DataFrame) -> dict[str, float]:
    """Return a few descriptive statistics that are helpful for context."""
    stats = {
        "records": len(df),
        "mean": df["avg_temp_c"].mean(),
        "median": df["avg_temp_c"].median(),
        "min_temp": df["avg_temp_c"].min(),
        "max_temp": df["avg_temp_c"].max(),
    }
    min_row = df.loc[df["avg_temp_c"].idxmin()]
    max_row = df.loc[df["avg_temp_c"].idxmax()]
    stats["coldest_date"] = min_row["date"].date()
    stats["warmest_date"] = max_row["date"].date()
    return stats


def plot_daily_profile(df: pd.DataFrame) -> None:
    """Plot the full daily time series."""
    mean_temp = df["avg_temp_c"].mean()
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(df["date"], df["avg_temp_c"], color="#1f77b4", linewidth=0.8)
    ax.axhline(mean_temp, color="black", linestyle="--", linewidth=1, label=f"Mean {mean_temp:.1f}℃")
    ax.set_ylabel("Average temperature (℃)")
    ax.set_title("Daily average temperature in Tokyo (2020-2025)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "1_daily_average.png", dpi=200)
    plt.close(fig)


def plot_monthly_cycle(df: pd.DataFrame) -> None:
    """Plot the seasonal cycle using monthly averages."""
    monthly = df.set_index("date")["avg_temp_c"].resample("ME").mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(monthly.index, monthly.values, marker="o", linewidth=1.2, color="#d62728")
    ax.set_ylabel("Average temperature (℃)")
    ax.set_title("Monthly mean temperature")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "2_monthly_means.png", dpi=200)
    plt.close(fig)


def plot_rolling_trend(df: pd.DataFrame, window: int = 30) -> None:
    """Plot a rolling average to highlight medium-term swings."""
    sorted_df = df.set_index("date")
    rolling = sorted_df["avg_temp_c"].rolling(window=window, min_periods=window // 2).mean()
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(sorted_df.index, sorted_df["avg_temp_c"], color="#9ecae1", linewidth=0.4, label="Daily")
    ax.plot(rolling.index, rolling.values, color="#08519c", linewidth=1.5, label=f"{window}-day rolling mean")
    ax.set_ylabel("Average temperature (℃)")
    ax.set_title(f"{window}-day rolling trend")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "3_rolling_trend.png", dpi=200)
    plt.close(fig)


def plot_distribution(df: pd.DataFrame) -> None:
    """Show how temperatures are distributed by month."""
    month_order = range(1, 13)
    data_by_month = [df.loc[df["month"] == m, "avg_temp_c"].values for m in month_order]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.boxplot(
        data_by_month,
        tick_labels=[str(m) for m in month_order],
        showfliers=False,
        patch_artist=True,
        boxprops=dict(facecolor="#ffbb78", alpha=0.7),
    )
    ax.set_xlabel("Month")
    ax.set_ylabel("Average temperature (℃)")
    ax.set_title("Monthly temperature distribution (daily data)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "4_monthly_boxplot.png", dpi=200)
    plt.close(fig)


def plot_yearly_overlay(df: pd.DataFrame) -> None:
    """Plot each year's seasonal curve with month/day on the x-axis."""
    df = df.copy()
    df["year"] = df["date"].dt.year
    # Use a leap year baseline so Feb 29 is representable
    df["month_day"] = pd.to_datetime("2000-" + df["date"].dt.strftime("%m-%d"))
    fig, ax = plt.subplots(figsize=(10, 4))
    years = sorted(df["year"].unique())
    cmap = plt.get_cmap("viridis", len(years))
    for idx, year in enumerate(years):
        yearly = df[df["year"] == year].sort_values("month_day")
        ax.plot(
            yearly["month_day"],
            yearly["avg_temp_c"],
            label=str(year),
            color=cmap(idx),
            linewidth=1,
        )
    ax.set_xlabel("Month-Day")
    ax.set_ylabel("Average temperature (℃)")
    ax.set_title("Seasonal temperature profile by year")
    ax.grid(alpha=0.3)
    ax.legend(ncol=2, fontsize="small", frameon=False)
    # Show ticks as month abbreviations
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "5_yearly_overlay.png", dpi=200)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    df = load_temperature_data(DATA_FILE)
    stats = summarize(df)

    print("---- Temperature overview ----")
    print(f"Records: {stats['records']}")
    print(f"Mean / median: {stats['mean']:.1f}℃ / {stats['median']:.1f}℃")
    print(f"Coldest day: {stats['coldest_date']} at {stats['min_temp']:.1f}℃")
    print(f"Warmest day: {stats['warmest_date']} at {stats['max_temp']:.1f}℃")

    plot_daily_profile(df)
    plot_monthly_cycle(df)
    plot_rolling_trend(df)
    plot_distribution(df)
    plot_yearly_overlay(df)
    print(f"Saved figures to {FIG_DIR.resolve()}")


if __name__ == "__main__":
    main()
