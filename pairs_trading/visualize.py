"""
Visualization module for pairs trading analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional


def set_style():
    """Set consistent plot styling."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.figsize": (14, 6),
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    })


def plot_correlation_heatmap(
    log_returns: pd.DataFrame,
    save_path: Optional[str] = None,
):
    """
    Plot a heatmap of pairwise return correlations.

    Parameters
    ----------
    log_returns : pd.DataFrame
        Log returns.
    save_path : str, optional
        File path to save the figure.
    """
    set_style()
    corr = log_returns.corr()

    fig, ax = plt.subplots(figsize=(16, 14))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
        center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
        ax=ax, annot_kws={"size": 8},
    )
    ax.set_title("ETF Return Correlation Matrix")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_pair_prices(
    prices: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    save_path: Optional[str] = None,
):
    """
    Plot normalized price series for a pair of ETFs.

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted close prices.
    ticker_a, ticker_b : str
        Pair tickers.
    save_path : str, optional
        File path to save the figure.
    """
    set_style()
    fig, ax = plt.subplots()

    # Normalize to 100 at start
    common_idx = prices[[ticker_a, ticker_b]].dropna().index
    norm_a = prices[ticker_a].loc[common_idx] / prices[ticker_a].loc[common_idx].iloc[0] * 100
    norm_b = prices[ticker_b].loc[common_idx] / prices[ticker_b].loc[common_idx].iloc[0] * 100

    ax.plot(norm_a, label=ticker_a, linewidth=1.5)
    ax.plot(norm_b, label=ticker_b, linewidth=1.5)
    ax.set_title(f"Normalized Prices: {ticker_a} vs {ticker_b}")
    ax.set_ylabel("Normalized Price (base=100)")
    ax.set_xlabel("Date")
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_spread_and_zscore(
    spread: pd.Series,
    zscore: pd.Series,
    ticker_a: str,
    ticker_b: str,
    theta_in: float = 2.0,
    theta_out: float = 1.0,
    save_path: Optional[str] = None,
):
    """
    Plot the spread and z-score with entry/exit threshold lines.

    Parameters
    ----------
    spread : pd.Series
        Spread time series.
    zscore : pd.Series
        Rolling z-score.
    ticker_a, ticker_b : str
        Pair tickers.
    theta_in : float
        Entry threshold.
    theta_out : float
        Exit threshold.
    save_path : str, optional
        File path to save the figure.
    """
    set_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Spread
    ax1.plot(spread, color="steelblue", linewidth=1)
    ax1.axhline(spread.mean(), color="red", linestyle="--", alpha=0.7, label="Mean")
    ax1.set_title(f"Spread: {ticker_a} - beta * {ticker_b}")
    ax1.set_ylabel("Spread")
    ax1.legend()

    # Z-score with thresholds
    ax2.plot(zscore, color="steelblue", linewidth=1)
    ax2.axhline(theta_in, color="red", linestyle="--", alpha=0.7, label=f"Entry (+{theta_in})")
    ax2.axhline(-theta_in, color="red", linestyle="--", alpha=0.7, label=f"Entry (-{theta_in})")
    ax2.axhline(theta_out, color="green", linestyle=":", alpha=0.7, label=f"Exit (+{theta_out})")
    ax2.axhline(-theta_out, color="green", linestyle=":", alpha=0.7, label=f"Exit (-{theta_out})")
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_title("Z-Score with Trading Thresholds")
    ax2.set_ylabel("Z-Score")
    ax2.set_xlabel("Date")
    ax2.legend(loc="upper right")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_signals_on_zscore(
    signals: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    theta_in: float = 2.0,
    theta_out: float = 1.0,
    save_path: Optional[str] = None,
):
    """
    Plot z-score with long/short entry and exit markers.

    Parameters
    ----------
    signals : pd.DataFrame
        Output from generate_signals (columns: zscore, signal, position).
    ticker_a, ticker_b : str
        Pair tickers.
    theta_in, theta_out : float
        Thresholds.
    save_path : str, optional
        File path to save the figure.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(signals["zscore"], color="steelblue", linewidth=1, label="Z-Score")
    ax.axhline(theta_in, color="red", linestyle="--", alpha=0.5)
    ax.axhline(-theta_in, color="red", linestyle="--", alpha=0.5)
    ax.axhline(theta_out, color="green", linestyle=":", alpha=0.5)
    ax.axhline(-theta_out, color="green", linestyle=":", alpha=0.5)
    ax.axhline(0, color="black", linewidth=0.5)

    # Mark entry/exit points
    pos_diff = signals["position"].diff()
    long_entries = signals.index[pos_diff == 1]
    short_entries = signals.index[pos_diff == -1]
    exits = signals.index[(pos_diff != 0) & (signals["position"] == 0)]

    ax.scatter(long_entries, signals.loc[long_entries, "zscore"],
               marker="^", color="green", s=80, zorder=5, label="Long Entry")
    ax.scatter(short_entries, signals.loc[short_entries, "zscore"],
               marker="v", color="red", s=80, zorder=5, label="Short Entry")
    ax.scatter(exits, signals.loc[exits, "zscore"],
               marker="x", color="black", s=60, zorder=5, label="Exit")

    ax.set_title(f"Trading Signals: {ticker_a}-{ticker_b}")
    ax.set_ylabel("Z-Score")
    ax.set_xlabel("Date")
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_cumulative_returns(
    trade_results: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    save_path: Optional[str] = None,
):
    """
    Plot cumulative return of the pairs trading strategy vs buy-and-hold.

    Parameters
    ----------
    trade_results : pd.DataFrame
        Output from compute_trade_returns.
    ticker_a, ticker_b : str
        Pair tickers.
    save_path : str, optional
        File path to save the figure.
    """
    set_style()
    fig, ax = plt.subplots()

    ax.plot(trade_results["cumulative_return"] * 100, color="steelblue",
            linewidth=2, label="Pairs Strategy")

    # Buy and hold for each ETF
    if "return_a" in trade_results.columns:
        bh_a = (1 + trade_results["return_a"].fillna(0)).cumprod() - 1
        bh_b = (1 + trade_results["return_b"].fillna(0)).cumprod() - 1
        ax.plot(bh_a * 100, color="green", linewidth=1, alpha=0.7, label=f"Buy & Hold {ticker_a}")
        ax.plot(bh_b * 100, color="red", linewidth=1, alpha=0.7, label=f"Buy & Hold {ticker_b}")

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title(f"Cumulative Returns: {ticker_a}-{ticker_b} Pairs Strategy")
    ax.set_ylabel("Cumulative Return (%)")
    ax.set_xlabel("Date")
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_optimization_surface(
    grid_results: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    value_col: str = "total_return",
    save_path: Optional[str] = None,
):
    """
    Plot a heatmap of a metric across theta_in and theta_out values.

    A broad plateau around the optimum indicates a robust parameter
    choice; a sharp peak or a boundary optimum indicates overfitting.

    Parameters
    ----------
    grid_results : pd.DataFrame
        Output from optimize_pair_thresholds (columns: theta_in,
        theta_out, total_return, sharpe, num_trades).
    ticker_a, ticker_b : str
        Pair tickers.
    value_col : str
        Which grid column to plot (e.g. "sharpe" or "total_return").
    save_path : str, optional
        File path to save the figure.
    """
    set_style()
    pivot = grid_results.pivot_table(
        values=value_col, index="theta_out", columns="theta_in", aggfunc="mean"
    )

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap="RdYlGn", center=0,
        ax=ax, linewidths=0.5, annot_kws={"size": 8},
    )
    ax.set_title(f"Parameter Sensitivity ({value_col}): {ticker_a}-{ticker_b}")
    ax.set_ylabel("Exit Threshold (theta_out)")
    ax.set_xlabel("Entry Threshold (theta_in)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_summary_table(
    summary_df: pd.DataFrame,
    save_path: Optional[str] = None,
):
    """
    Display a formatted summary table of backtest results.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Output from summarize_results.
    save_path : str, optional
        File path to save the figure.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(14, max(4, len(summary_df) * 0.5 + 2)))
    ax.axis("off")

    # Format numeric columns
    display_df = summary_df.copy()
    for col in ["avg_return", "std_return", "avg_max_drawdown",
                "total_return", "ann_return", "ann_vol", "max_drawdown"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].map("{:.4f}".format)
    for col in ["avg_sharpe", "sharpe", "t_stat", "selection_rate"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].map("{:.2f}".format)
    for col in ["avg_trades_per_window"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].map("{:.1f}".format)

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    ax.set_title("Pairs Trading Backtest Summary", fontsize=14, pad=20)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_portfolio_returns(
    portfolio_returns: pd.Series,
    holdout_start: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """
    Plot the portfolio cumulative return with an optional marker at the
    start of the hold-out period.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Daily portfolio returns (from portfolio.build_portfolio).
    holdout_start : str, optional
        Date (YYYY-MM-DD) where the hold-out period begins.
    save_path : str, optional
        File path to save the figure.
    """
    set_style()
    fig, ax = plt.subplots()

    cumulative = (1 + portfolio_returns.fillna(0)).cumprod() - 1
    ax.plot(cumulative * 100, color="steelblue", linewidth=1.5,
            label="Pairs Portfolio (1/N, dedup)")

    if holdout_start is not None:
        cutoff = pd.Timestamp(holdout_start)
        if cumulative.index.min() < cutoff < cumulative.index.max():
            ax.axvline(cutoff, color="red", linestyle="--", alpha=0.7,
                       label=f"Hold-out start ({holdout_start})")

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Portfolio Cumulative Return")
    ax.set_ylabel("Cumulative Return (%)")
    ax.set_xlabel("Date")
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
