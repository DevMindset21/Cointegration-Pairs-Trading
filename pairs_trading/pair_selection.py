"""
Pair selection module: correlation filtering and cointegration testing.
"""

import pandas as pd
import numpy as np
from itertools import combinations
from statsmodels.tsa.stattools import coint
from typing import List, Tuple


def compute_correlation_matrix(log_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pairwise Pearson correlation matrix of log returns.

    Parameters
    ----------
    log_returns : pd.DataFrame
        Log returns with tickers as columns.

    Returns
    -------
    pd.DataFrame
        Correlation matrix.
    """
    return log_returns.corr()


def filter_by_correlation(
    log_returns: pd.DataFrame,
    threshold: float = 0.8,
    verbose: bool = True,
) -> List[Tuple[str, str, float]]:
    """
    Find all ETF pairs with correlation above the threshold.

    Parameters
    ----------
    log_returns : pd.DataFrame
        Log returns.
    threshold : float
        Minimum correlation to keep a pair.
    verbose : bool
        Print the number of pairs found.

    Returns
    -------
    list of (ticker_a, ticker_b, correlation)
        Pairs sorted by correlation descending.
    """
    corr_matrix = compute_correlation_matrix(log_returns)
    tickers = corr_matrix.columns.tolist()
    pairs = []

    for i, j in combinations(range(len(tickers)), 2):
        corr_val = corr_matrix.iloc[i, j]
        if corr_val >= threshold:
            pairs.append((tickers[i], tickers[j], corr_val))

    pairs.sort(key=lambda x: x[2], reverse=True)
    if verbose:
        print(f"Found {len(pairs)} pairs with correlation >= {threshold}")
    return pairs


def test_cointegration(
    prices: pd.DataFrame,
    pair: Tuple[str, str],
) -> Tuple[float, float]:
    """
    Run the Engle-Granger cointegration test on a pair of price series.

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted close prices.
    pair : tuple of (str, str)
        Ticker symbols for the two ETFs.

    Returns
    -------
    tuple of (t_stat, p_value)
        Test statistic and p-value from the cointegration test.
    """
    series_a = prices[pair[0]].dropna()
    series_b = prices[pair[1]].dropna()

    # Align the two series on common dates
    common_idx = series_a.index.intersection(series_b.index)
    series_a = series_a.loc[common_idx]
    series_b = series_b.loc[common_idx]

    t_stat, p_value, _ = coint(series_a, series_b)
    return t_stat, p_value


def filter_by_cointegration(
    prices: pd.DataFrame,
    correlated_pairs: List[Tuple[str, str, float]],
    pvalue_threshold: float = 0.05,
    verbose: bool = True,
) -> List[dict]:
    """
    Filter correlated pairs by the Engle-Granger cointegration test.

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted close prices.
    correlated_pairs : list of (ticker_a, ticker_b, correlation)
        Pairs that passed the correlation filter.
    pvalue_threshold : float
        Maximum p-value to accept cointegration.

    Returns
    -------
    list of dict
        Each dict has keys: ticker_a, ticker_b, correlation, t_stat, p_value.
    """
    cointegrated = []

    for ticker_a, ticker_b, corr_val in correlated_pairs:
        try:
            t_stat, p_value = test_cointegration(prices, (ticker_a, ticker_b))
            if p_value < pvalue_threshold:
                cointegrated.append({
                    "ticker_a": ticker_a,
                    "ticker_b": ticker_b,
                    "correlation": corr_val,
                    "t_stat": t_stat,
                    "p_value": p_value,
                })
        except Exception as e:
            if verbose:
                print(f"Cointegration test failed for {ticker_a}-{ticker_b}: {e}")

    cointegrated.sort(key=lambda x: x["p_value"])
    if verbose:
        print(f"Found {len(cointegrated)} cointegrated pairs (p-value < {pvalue_threshold})")
    return cointegrated


def select_pairs(
    prices: pd.DataFrame,
    log_returns: pd.DataFrame,
    correlation_threshold: float = 0.8,
    cointegration_pvalue: float = 0.05,
) -> List[dict]:
    """
    Full pair selection pipeline: correlation filter then cointegration test.

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted close prices.
    log_returns : pd.DataFrame
        Log returns.
    correlation_threshold : float
        Min correlation for initial filtering.
    cointegration_pvalue : float
        Max p-value for cointegration test.

    Returns
    -------
    list of dict
        Selected cointegrated pairs with metadata.
    """
    correlated = filter_by_correlation(log_returns, correlation_threshold)
    cointegrated = filter_by_cointegration(prices, correlated, cointegration_pvalue)
    return cointegrated


def select_pairs_in_window(
    prices: pd.DataFrame,
    start: int,
    end: int,
    correlation_threshold: float = 0.8,
    cointegration_pvalue: float = 0.05,
    candidates: List[Tuple[str, str]] = None,
    min_coverage: float = 0.9,
) -> List[dict]:
    """
    Point-in-time pair selection using ONLY data inside [start, end).

    Used by the rolling backtest so that pair selection at each window
    sees no future data (avoids the look-ahead bias of selecting pairs
    on the full sample and then trading them over that same sample).

    Parameters
    ----------
    prices : pd.DataFrame
        Full price DataFrame (sliced internally by iloc).
    start, end : int
        iloc bounds of the formation window.
    correlation_threshold : float
        Min return correlation.
    cointegration_pvalue : float
        Max Engle-Granger p-value.
    candidates : list of (ticker_a, ticker_b), optional
        Restrict the search to these pairs. None = all combinations.
    min_coverage : float
        Minimum fraction of non-NaN observations in the window required
        to consider a ticker (drops not-yet-launched ETFs).

    Returns
    -------
    list of dict
        Same schema as select_pairs (ticker_a, ticker_b, correlation,
        t_stat, p_value).
    """
    window_prices = prices.iloc[start:end]

    # Drop tickers without enough data in this window
    coverage = window_prices.notna().mean()
    valid = coverage[coverage >= min_coverage].index
    window_prices = window_prices[valid]
    if window_prices.shape[1] < 2:
        return []

    window_returns = np.log(window_prices / window_prices.shift(1)).dropna(how="all")

    correlated = filter_by_correlation(
        window_returns, correlation_threshold, verbose=False
    )
    if candidates is not None:
        allowed = {frozenset(c) for c in candidates}
        correlated = [
            p for p in correlated if frozenset((p[0], p[1])) in allowed
        ]

    return filter_by_cointegration(
        window_prices, correlated, cointegration_pvalue, verbose=False
    )


def profit_to_cost_ratio(
    prices: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    beta: float,
    alpha: float = 0.0,
    formation_start: int = None,
    formation_end: int = None,
    theta_in: float = 0.0,
    theta_out: float = 0.0,
    transaction_cost: float = 0.0,
) -> float:
    """
    Expected profit per round trip relative to round-trip cost.

    A trade captures roughly (theta_in - theta_out) standard deviations
    of the spread; expressed as a fraction of the capital deployed on
    both legs, this must comfortably exceed the round-trip transaction
    cost or the pair is a structural money-loser (e.g. AGG-BND, whose
    spread moves a few bps while costs are 20 bps round trip).

    Returns
    -------
    float
        Expected reversion profit per trade divided by round-trip cost.
        Trade only if this is >= ~3.
    """
    a = prices[ticker_a].iloc[formation_start:formation_end]
    b = prices[ticker_b].iloc[formation_start:formation_end]
    spread = (a - alpha - beta * b).dropna()
    if len(spread) < 2:
        return 0.0

    sigma = spread.std()
    gross_notional = (a + abs(beta) * b).dropna().iloc[-1]
    expected_profit = (theta_in - theta_out) * sigma / gross_notional
    round_trip_cost = 2 * transaction_cost
    return expected_profit / round_trip_cost if round_trip_cost > 0 else np.inf
