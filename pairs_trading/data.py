"""
Data pipeline for downloading and preprocessing ETF price data.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Optional


# Default ETF universe from the paper
DEFAULT_ETFS = [
    "SPY", "IVV", "VOO", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLP",
    "XLU", "XLB", "XLY", "XLRE", "IWM", "IWF", "IWD", "EFA", "EEM", "VTI",
    "VEA", "VWO", "AGG", "BND", "TLT", "GLD", "SLV", "DIA", "MDY", "IJH",
]


def download_etf_data(
    tickers: List[str] = None,
    start_date: str = "2000-01-01",
    end_date: str = "2024-12-31",
) -> pd.DataFrame:
    """
    Download adjusted close prices for a list of ETF tickers.

    Parameters
    ----------
    tickers : list of str
        ETF ticker symbols. Defaults to DEFAULT_ETFS.
    start_date : str
        Start date in YYYY-MM-DD format.
    end_date : str
        End date in YYYY-MM-DD format.

    Returns
    -------
    pd.DataFrame
        DataFrame with dates as index and tickers as columns (adjusted close).
    """
    if tickers is None:
        tickers = DEFAULT_ETFS

    print(f"Downloading data for {len(tickers)} ETFs from {start_date} to {end_date}...")
    raw = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)

    # yfinance returns MultiIndex columns when multiple tickers are passed
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
        prices.columns = tickers

    # Drop ETFs that have no data at all
    prices = prices.dropna(axis=1, how="all")

    dropped = set(tickers) - set(prices.columns)
    if dropped:
        print(f"Warning: dropped {dropped} (no data available)")

    print(f"Downloaded {prices.shape[1]} ETFs, {prices.shape[0]} trading days")
    return prices


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute log returns from a price DataFrame.

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted close prices.

    Returns
    -------
    pd.DataFrame
        Log returns (first row is NaN and is dropped).
    """
    log_returns = np.log(prices / prices.shift(1)).dropna()
    return log_returns


def preprocess(
    prices: pd.DataFrame,
    min_history: int = 252,
) -> pd.DataFrame:
    """
    Clean price data: forward-fill gaps, drop ETFs with insufficient history.

    Parameters
    ----------
    prices : pd.DataFrame
        Raw adjusted close prices.
    min_history : int
        Minimum number of non-NaN trading days required to keep an ETF.

    Returns
    -------
    pd.DataFrame
        Cleaned price DataFrame.
    """
    # Forward-fill small gaps (weekends already excluded by yfinance)
    prices = prices.ffill()

    # Drop ETFs with insufficient history
    valid_counts = prices.notna().sum()
    keep = valid_counts[valid_counts >= min_history].index
    prices = prices[keep]

    # Drop any remaining rows where all values are NaN
    prices = prices.dropna(how="all")

    print(f"After preprocessing: {prices.shape[1]} ETFs, {prices.shape[0]} trading days")
    return prices


def get_data(
    tickers: Optional[List[str]] = None,
    start_date: str = "2000-01-01",
    end_date: str = "2024-12-31",
    min_history: int = 252,
) -> tuple:
    """
    Full data pipeline: download, preprocess, compute returns.

    Returns
    -------
    tuple of (prices, log_returns)
        prices : pd.DataFrame — cleaned adjusted close prices
        log_returns : pd.DataFrame — log returns
    """
    prices = download_etf_data(tickers, start_date, end_date)
    prices = preprocess(prices, min_history)
    log_returns = compute_log_returns(prices)
    return prices, log_returns
