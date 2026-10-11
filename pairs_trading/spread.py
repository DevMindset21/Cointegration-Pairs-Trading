"""
Spread and Z-score computation for cointegrated pairs.
"""

import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant


def compute_hedge_ratio(
    prices: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    start: int = None,
    end: int = None,
) -> tuple[float, float]:
    """
    Estimate the hedge ratio and intercept via OLS regression:
    A = alpha + beta * B.

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted close prices.
    ticker_a : str
        Dependent variable ticker.
    ticker_b : str
        Independent variable ticker.
    start : int, optional
        Start index (iloc) for the estimation window.
    end : int, optional
        End index (iloc) for the estimation window.

    Returns
    -------
    tuple[float, float]
        OLS hedge ratio (beta) and intercept (alpha).
    """
    if start is not None and end is not None:
        a = prices[ticker_a].iloc[start:end].values
        b = prices[ticker_b].iloc[start:end].values
    else:
        a = prices[ticker_a].values
        b = prices[ticker_b].values

    # Drop NaNs from both series
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]

    b_const = add_constant(b)
    model = OLS(a, b_const).fit()
    alpha = model.params[0]
    beta = model.params[1]
    return beta, alpha


def compute_spread(
    prices: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    beta: float,
    alpha: float,
) -> pd.Series:
    """
    Compute the spread: Z_t = Price_A - alpha - beta * Price_B.

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted close prices.
    ticker_a : str
        First ticker.
    ticker_b : str
        Second ticker.
    beta : float
        Hedge ratio from OLS regression.
    alpha : float
        Intercept from the OLS regression.

    Returns
    -------
    pd.Series
        Spread time series.
    """
    spread = prices[ticker_a] - alpha - beta * prices[ticker_b]
    spread.name = f"{ticker_a}-{ticker_b}_spread"
    return spread


def compute_zscore(
    spread: pd.Series,
    window: int = 200,
) -> pd.Series:
    """
    Compute the rolling z-score of the spread.

    z_t = (spread_t - rolling_mean) / rolling_std

    Parameters
    ----------
    spread : pd.Series
        Spread time series.
    window : int
        Rolling window size in trading days.

    Returns
    -------
    pd.Series
        Rolling z-score.
    """
    rolling_mean = spread.rolling(window=window).mean()
    rolling_std = spread.rolling(window=window).std()
    zscore = (spread - rolling_mean) / rolling_std
    zscore.name = f"{spread.name}_zscore"
    return zscore


def estimate_half_life(spread: pd.Series) -> float:
    """
    Estimate the mean-reversion half-life of a spread in trading days
    via an AR(1) / discrete Ornstein-Uhlenbeck fit:

        dS_t = a + k * S_{t-1} + eps

    half_life = -ln(2) / ln(1 + k), valid when k < 0 (mean-reverting).

    Returns
    -------
    float
        Half-life in trading days; np.inf if the spread shows no
        mean reversion (k >= 0) or there is insufficient data.
    """
    s = spread.dropna()
    if len(s) < 20:
        return np.inf

    ds = s.diff().dropna()
    lag = s.shift(1).loc[ds.index]

    X = add_constant(lag.values)
    model = OLS(ds.values, X).fit()
    k = model.params[1]

    if k >= 0 or (1 + k) <= 0:
        return np.inf
    return float(-np.log(2) / np.log(1 + k))


def compute_pair_spread_and_zscore(
    prices: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    formation_start: int = None,
    formation_end: int = None,
    rolling_window: int = 200,
) -> dict:
    """
    Full spread pipeline for a single pair: hedge ratio, spread, z-score.

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted close prices.
    ticker_a : str
        First ETF ticker.
    ticker_b : str
        Second ETF ticker.
    formation_start : int, optional
        Start iloc for hedge ratio estimation.
    formation_end : int, optional
        End iloc for hedge ratio estimation.
    rolling_window : int
        Window for rolling z-score.

    Returns
    -------
    dict with keys: beta, spread, zscore
    """
    beta, alpha = compute_hedge_ratio(
        prices, ticker_a, ticker_b, formation_start, formation_end,
    )
    spread = compute_spread(prices, ticker_a, ticker_b, beta, alpha)
    zscore = compute_zscore(spread, rolling_window)

    return {
        "beta": beta,
        "alpha": alpha,
        "spread": spread,
        "zscore": zscore,
    }
