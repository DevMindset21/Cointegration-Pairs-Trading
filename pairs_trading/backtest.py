"""
Backtesting engine with Point-In-Time (PIT) methodology.

PIT applies to BOTH pair selection and hedge-ratio estimation: at each
rolling window, pairs are re-selected using only formation-window data,
so no future information leaks into which pairs get traded.
"""

import pandas as pd
import numpy as np
from typing import List, Optional

from .spread import compute_hedge_ratio, compute_spread, compute_zscore, estimate_half_life
from .strategy import generate_signals, compute_trade_returns
from .pair_selection import select_pairs_in_window, profit_to_cost_ratio


def compute_performance_metrics(returns: pd.Series) -> dict:
    """
    Compute standard performance metrics for a daily return series.

    Sharpe is mean(daily) / std(daily) * sqrt(252) — no compounding of
    short-window returns to annual, which explodes on 63-day windows.
    t_stat = sharpe * sqrt(years) gives a rough significance measure.

    Parameters
    ----------
    returns : pd.Series
        Daily strategy returns.

    Returns
    -------
    dict
        Performance metrics including Sharpe, t-stat, max drawdown, etc.
    """
    returns = returns.dropna()
    if len(returns) == 0:
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "t_stat": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "num_trading_days": 0,
        }

    cumulative = (1 + returns).cumprod()
    total_return = cumulative.iloc[-1] - 1
    n_days = len(returns)

    mean_daily = returns.mean()
    std_daily = returns.std()
    annualized_return = mean_daily * 252
    annualized_vol = std_daily * np.sqrt(252)
    sharpe = mean_daily / std_daily * np.sqrt(252) if std_daily > 0 else 0.0
    t_stat = sharpe * np.sqrt(n_days / 252)

    # Max drawdown
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    # Win rate
    winning_days = (returns > 0).sum()
    total_days = (returns != 0).sum()
    win_rate = winning_days / total_days if total_days > 0 else 0.0

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_vol,
        "sharpe_ratio": sharpe,
        "t_stat": t_stat,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "num_trading_days": n_days,
    }


def count_trades(positions: pd.Series) -> int:
    """
    Count trades as position entries: flat -> long/short transitions,
    plus direct flips (long <-> short), which are one exit + one entry
    but counted as a single new trade.
    """
    pos = positions.fillna(0)
    prev = pos.shift(1).fillna(0)
    entries = ((prev == 0) & (pos != 0)) | (prev * pos < 0)
    return int(entries.sum())


def backtest_pair(
    prices: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    formation_start: int,
    formation_end: int,
    trading_start: int,
    trading_end: int,
    theta_in: float = 2.0,
    theta_out: float = 1.0,
    rolling_window: int = 200,
    transaction_cost: float = 0.001,
    stop_loss: float = None,
    max_holding: int = None,
    target_vol: float = None,
    vol_window: int = 60,
    max_leverage: float = 3.0,
) -> dict:
    """
    Backtest a single pair over a specific formation + trading window.

    Point-In-Time methodology:
    - Formation period: estimate hedge ratio and pair parameters
    - Trading period: generate signals and compute returns (out-of-sample)
    - Positions are force-closed on the last day of the trading window so
      exit costs are paid and P&L is fully realized.

    Parameters
    ----------
    prices : pd.DataFrame
        Full price DataFrame.
    ticker_a, ticker_b : str
        Pair tickers.
    formation_start, formation_end : int
        iloc indices for the formation (training) window.
    trading_start, trading_end : int
        iloc indices for the trading (test) window.
    theta_in : float
        Z-score entry threshold.
    theta_out : float
        Z-score exit threshold.
    rolling_window : int
        Rolling window for z-score.
    transaction_cost : float
        Cost per trade.
    stop_loss : float, optional
        Z-score stop-loss (see generate_signals).
    max_holding : int, optional
        Time stop in days (see generate_signals).
    target_vol : float, optional
        Daily vol target for position sizing (see compute_trade_returns).
    vol_window : int
        Rolling window for the sizing vol estimate.
    max_leverage : float
        Leverage cap for vol targeting.

    Returns
    -------
    dict with keys: ticker_a, ticker_b, beta, metrics, trade_results
    """
    # Estimate hedge ratio on formation period only (PIT)
    beta = compute_hedge_ratio(prices, ticker_a, ticker_b, formation_start, formation_end)

    # Compute spread over the full range needed for rolling z-score
    # We need rolling_window days before trading_start for the z-score to be valid
    spread_start = max(0, trading_start - rolling_window)
    spread_prices = prices.iloc[spread_start:trading_end]
    spread = compute_spread(spread_prices, ticker_a, ticker_b, beta)
    zscore = compute_zscore(spread, rolling_window)

    # Restrict signals and returns to trading period only
    trading_prices = prices.iloc[trading_start:trading_end]
    trading_zscore = zscore.loc[trading_prices.index]

    signals = generate_signals(
        trading_zscore, theta_in, theta_out,
        stop_loss=stop_loss, max_holding=max_holding, force_close_end=True,
    )
    trade_results = compute_trade_returns(
        prices, ticker_a, ticker_b, signals, beta, transaction_cost,
        target_vol=target_vol, vol_window=vol_window, max_leverage=max_leverage,
    )

    metrics = compute_performance_metrics(trade_results["strategy_return"])
    metrics["num_trades"] = count_trades(trade_results["position"])

    return {
        "ticker_a": ticker_a,
        "ticker_b": ticker_b,
        "beta": beta,
        "theta_in": theta_in,
        "theta_out": theta_out,
        "metrics": metrics,
        "trade_results": trade_results,
        "signals": signals,
    }


def rolling_backtest(
    prices: pd.DataFrame,
    candidate_pairs: Optional[List[dict]] = None,
    formation_period: int = 252,
    trading_period: int = 63,
    theta_in: float = 2.0,
    theta_out: float = 1.0,
    rolling_window: int = 200,
    transaction_cost: float = 0.001,
    pit_selection: bool = True,
    correlation_threshold: float = 0.8,
    cointegration_pvalue: float = 0.05,
    min_profit_cost_ratio: float = 3.0,
    require_persistence: bool = False,
    half_life_range: tuple = None,
    stop_loss: float = None,
    max_holding: int = None,
    target_vol: float = None,
    vol_window: int = 60,
    max_leverage: float = 3.0,
    selection_by_window: Optional[dict] = None,
) -> dict:
    """
    Run a rolling-window backtest with point-in-time pair selection.

    At each window:
    1. Re-select pairs using ONLY the formation window (correlation +
       Engle-Granger), unless pit_selection=False.
    2. Optionally require the pair to also have been selected in the
       PREVIOUS window (require_persistence) — a defense against
       cointegration-test false positives, which rarely persist.
    3. Optionally require the formation-window spread half-life to fall
       inside half_life_range (days) — reversion speed must match the
       trading horizon.
    4. Skip pairs whose expected reversion profit per trade is below
       min_profit_cost_ratio x round-trip cost.
    5. Backtest each surviving pair out-of-sample on the trading window.

    Parameters
    ----------
    prices : pd.DataFrame
        Full price DataFrame.
    candidate_pairs : list of dict, optional
        Each dict has 'ticker_a' and 'ticker_b'. Restricts the pair
        universe; None considers all combinations (PIT mode only).
    formation_period, trading_period : int
        Window sizes in trading days.
    theta_in, theta_out : float
        Z-score thresholds.
    rolling_window : int
        Rolling window for z-score.
    transaction_cost : float
        Cost per trade.
    pit_selection : bool
        Re-select pairs each window from formation data only.
    correlation_threshold, cointegration_pvalue : float
        PIT selection thresholds.
    min_profit_cost_ratio : float
        Minimum expected-profit-to-cost ratio to trade a pair in a
        window (0 disables the filter).
    stop_loss, max_holding, target_vol, vol_window, max_leverage
        Risk-management parameters passed to backtest_pair.
    selection_by_window : dict, optional
        Precomputed {formation_start: [pair dicts]} from
        precompute_selections(). Selection depends only on the formation
        window and the correlation/cointegration thresholds — never on
        theta, costs, or risk parameters — so it can be computed once and
        reused across a parameter or cost sweep.

    Returns
    -------
    dict with keys:
        results : list of dict — per-pair {ticker_a, ticker_b, windows}
        n_windows : int — total rolling windows evaluated
        selection_counts : pd.Series — windows in which each pair was
            selected (diagnostic for selection stability)
        cost_filtered : pd.Series — windows in which each pair was
            selected but skipped by the cost filter
    """
    n = len(prices)
    results_by_pair = {}
    selection_counts = {}
    cost_filtered = {}
    n_windows = 0
    prev_selected = set()

    if candidate_pairs is not None:
        candidate_tuples = [(p["ticker_a"], p["ticker_b"]) for p in candidate_pairs]
    else:
        candidate_tuples = None

    start = 0
    while start + formation_period + trading_period <= n:
        formation_start = start
        formation_end = start + formation_period
        trading_start = formation_end
        trading_end = min(trading_start + trading_period, n)
        n_windows += 1

        if selection_by_window is not None:
            window_pairs = selection_by_window.get(formation_start, [])
        elif pit_selection:
            window_pairs = select_pairs_in_window(
                prices, formation_start, formation_end,
                correlation_threshold, cointegration_pvalue,
                candidates=candidate_tuples,
            )
        else:
            window_pairs = candidate_pairs or []

        current_selected = {
            f"{p['ticker_a']}-{p['ticker_b']}" for p in window_pairs
        }

        for pair in window_pairs:
            ticker_a = pair["ticker_a"]
            ticker_b = pair["ticker_b"]
            key = f"{ticker_a}-{ticker_b}"
            selection_counts[key] = selection_counts.get(key, 0) + 1

            if require_persistence and key not in prev_selected:
                continue

            try:
                beta = compute_hedge_ratio(
                    prices, ticker_a, ticker_b, formation_start, formation_end
                )

                if half_life_range is not None:
                    formation_prices = prices.iloc[formation_start:formation_end]
                    spread = compute_spread(formation_prices, ticker_a, ticker_b, beta)
                    hl = estimate_half_life(spread)
                    if not (half_life_range[0] <= hl <= half_life_range[1]):
                        continue

                if min_profit_cost_ratio > 0:
                    ratio = profit_to_cost_ratio(
                        prices, ticker_a, ticker_b, beta,
                        formation_start, formation_end,
                        theta_in, theta_out, transaction_cost,
                    )
                    if ratio < min_profit_cost_ratio:
                        cost_filtered[key] = cost_filtered.get(key, 0) + 1
                        continue

                result = backtest_pair(
                    prices, ticker_a, ticker_b,
                    formation_start, formation_end,
                    trading_start, trading_end,
                    theta_in, theta_out,
                    rolling_window, transaction_cost,
                    stop_loss=stop_loss, max_holding=max_holding,
                    target_vol=target_vol, vol_window=vol_window,
                    max_leverage=max_leverage,
                )
                result["formation_dates"] = (
                    prices.index[formation_start].strftime("%Y-%m-%d"),
                    prices.index[formation_end - 1].strftime("%Y-%m-%d"),
                )
                result["trading_dates"] = (
                    prices.index[trading_start].strftime("%Y-%m-%d"),
                    prices.index[trading_end - 1].strftime("%Y-%m-%d"),
                )

                if key not in results_by_pair:
                    results_by_pair[key] = {
                        "ticker_a": ticker_a,
                        "ticker_b": ticker_b,
                        "windows": [],
                    }
                results_by_pair[key]["windows"].append(result)
            except Exception as e:
                print(f"Error backtesting {ticker_a}-{ticker_b} "
                      f"window {start}: {e}")

        prev_selected = current_selected
        start += trading_period  # roll forward by trading_period

    all_results = list(results_by_pair.values())
    print(f"Backtested {len(all_results)} pairs across {n_windows} rolling windows "
          f"(PIT selection={'on' if pit_selection else 'off'})")

    return {
        "results": all_results,
        "n_windows": n_windows,
        "selection_counts": pd.Series(selection_counts).sort_values(ascending=False),
        "cost_filtered": pd.Series(cost_filtered).sort_values(ascending=False)
        if cost_filtered else pd.Series(dtype=int),
    }


def precompute_selections(
    prices: pd.DataFrame,
    formation_period: int = 252,
    trading_period: int = 63,
    correlation_threshold: float = 0.8,
    cointegration_pvalue: float = 0.05,
) -> dict:
    """
    Run point-in-time pair selection once for every rolling window.

    Selection is a pure function of the formation window and the
    correlation/cointegration thresholds, so cost or parameter sweeps can
    reuse the result instead of repeating tens of thousands of
    Engle-Granger tests per configuration.

    Returns
    -------
    dict
        {formation_start (iloc) : list of selected pair dicts}
    """
    n = len(prices)
    selections = {}
    start = 0
    while start + formation_period + trading_period <= n:
        selections[start] = select_pairs_in_window(
            prices, start, start + formation_period,
            correlation_threshold, cointegration_pvalue,
        )
        start += trading_period
    return selections


def stitch_pair_returns(pair_result: dict) -> pd.Series:
    """
    Concatenate a pair's daily strategy returns across all trading
    windows into one continuous series (windows are non-overlapping).
    """
    parts = [w["trade_results"]["strategy_return"] for w in pair_result["windows"]]
    stitched = pd.concat(parts).sort_index()
    return stitched[~stitched.index.duplicated(keep="first")]


def summarize_results(all_results: List[dict], n_windows: int = None) -> pd.DataFrame:
    """
    Summarize rolling backtest results into a DataFrame.

    Each pair's daily returns are stitched across windows and metrics
    are computed ONCE on the full series — per-window Sharpe averaging
    is statistically meaningless on 63-day samples.

    Parameters
    ----------
    all_results : list of dict
        The 'results' list from rolling_backtest.
    n_windows : int, optional
        Total rolling windows, for the selection_rate column.

    Returns
    -------
    pd.DataFrame
        Summary with one row per pair.
    """
    rows = []
    for pair_result in all_results:
        ticker_a = pair_result["ticker_a"]
        ticker_b = pair_result["ticker_b"]
        windows = pair_result["windows"]

        stitched = stitch_pair_returns(pair_result)
        metrics = compute_performance_metrics(stitched)
        num_trades = sum(w["metrics"]["num_trades"] for w in windows)

        row = {
            "pair": f"{ticker_a}-{ticker_b}",
            "num_windows": len(windows),
            "total_return": metrics["total_return"],
            "ann_return": metrics["annualized_return"],
            "ann_vol": metrics["annualized_volatility"],
            "sharpe": metrics["sharpe_ratio"],
            "t_stat": metrics["t_stat"],
            "max_drawdown": metrics["max_drawdown"],
            "total_trades": num_trades,
        }
        if n_windows:
            row["selection_rate"] = len(windows) / n_windows
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("sharpe", ascending=False).reset_index(drop=True)
    return df
