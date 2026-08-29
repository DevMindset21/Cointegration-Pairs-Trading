"""
Walk-forward parameter optimization over entry/exit thresholds.

Replaces the previous in-sample comparison (whose "test set" was a
subset of the optimization training window) with a proper walk-forward:
optimize on a trailing training window, then evaluate on the following
never-seen trading window. The objective is Sharpe (risk-adjusted), not
raw return, with a minimum-trades constraint so the optimizer cannot
win by near-inactivity.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple

from .spread import compute_hedge_ratio, compute_spread, compute_zscore
from .strategy import generate_signals, compute_trade_returns
from .backtest import compute_performance_metrics, count_trades, backtest_pair


def optimize_pair_thresholds(
    prices: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    formation_start: int,
    formation_end: int,
    training_start: int,
    training_end: int,
    rolling_window: int = 200,
    transaction_cost: float = 0.001,
    theta_in_range: Tuple[float, float, float] = (0.5, 3.0, 0.25),
    theta_out_range: Tuple[float, float, float] = (0.1, 1.5, 0.1),
    min_trades: int = 3,
    stop_loss: float = None,
    max_holding: int = None,
    target_vol: float = None,
    vol_window: int = 60,
    max_leverage: float = 3.0,
) -> dict:
    """
    Grid search over theta_in and theta_out on the training window.

    Objective: Sharpe ratio, subject to at least min_trades trades in
    training (otherwise the combo is not considered a valid optimum).
    The same risk overlays (stops, vol sizing) used in deployment are
    applied during optimization so training matches live behavior.

    Returns
    -------
    dict with keys:
        best_theta_in, best_theta_out, best_sharpe, beta, grid_results
        best_* are None when no combo satisfies the min-trades
        constraint (caller should fall back to defaults).
    """
    # Estimate hedge ratio on formation period
    beta = compute_hedge_ratio(prices, ticker_a, ticker_b, formation_start, formation_end)

    # Compute spread and z-score over the range needed
    spread_start = max(0, training_start - rolling_window)
    spread_prices = prices.iloc[spread_start:training_end]
    spread = compute_spread(spread_prices, ticker_a, ticker_b, beta)
    zscore = compute_zscore(spread, rolling_window)

    # Restrict to training window
    training_prices = prices.iloc[training_start:training_end]
    training_zscore = zscore.loc[training_prices.index]

    # Grid search
    theta_in_values = np.arange(*theta_in_range)
    theta_out_values = np.arange(*theta_out_range)

    best_sharpe = -np.inf
    best_theta_in = None
    best_theta_out = None
    grid_results = []

    for t_in in theta_in_values:
        for t_out in theta_out_values:
            if t_out >= t_in:
                continue  # exit threshold must be less than entry

            signals = generate_signals(
                training_zscore, t_in, t_out,
                stop_loss=stop_loss, max_holding=max_holding,
            )
            trade_results = compute_trade_returns(
                prices, ticker_a, ticker_b, signals, beta, transaction_cost,
                target_vol=target_vol, vol_window=vol_window,
                max_leverage=max_leverage,
            )

            metrics = compute_performance_metrics(trade_results["strategy_return"])
            n_trades = count_trades(trade_results["position"])

            grid_results.append({
                "theta_in": t_in,
                "theta_out": t_out,
                "total_return": metrics["total_return"],
                "sharpe": metrics["sharpe_ratio"],
                "num_trades": n_trades,
            })

            if n_trades >= min_trades and metrics["sharpe_ratio"] > best_sharpe:
                best_sharpe = metrics["sharpe_ratio"]
                best_theta_in = t_in
                best_theta_out = t_out

    return {
        "best_theta_in": best_theta_in,
        "best_theta_out": best_theta_out,
        "best_sharpe": best_sharpe if best_theta_in is not None else None,
        "beta": beta,
        "grid_results": pd.DataFrame(grid_results),
    }


def walk_forward_optimize(
    prices: pd.DataFrame,
    pairs: List[dict],
    formation_period: int = 252,
    training_period: int = 252,
    trading_period: int = 63,
    rolling_window: int = 200,
    transaction_cost: float = 0.001,
    theta_in_range: Tuple[float, float, float] = (0.5, 3.0, 0.25),
    theta_out_range: Tuple[float, float, float] = (0.1, 1.5, 0.1),
    default_theta_in: float = 2.0,
    default_theta_out: float = 1.0,
    min_trades: int = 3,
    stop_loss: float = None,
    max_holding: int = None,
    target_vol: float = None,
    vol_window: int = 60,
    max_leverage: float = 3.0,
) -> List[dict]:
    """
    Walk-forward optimization: at each rolling step, optimize thresholds
    on the trailing training window, then evaluate BOTH the optimized
    and the default thresholds on the following out-of-sample trading
    window. Training and test never overlap.

    Window layout per step (iloc):
        [t-training-formation, t-training) : formation for training beta
        [t-training, t)                    : training (grid search)
        [t-formation, t)                   : formation for test beta
        [t, t+trading)                     : out-of-sample test

    Returns
    -------
    list of dict, one per pair, with keys:
        ticker_a, ticker_b, n_windows,
        default_returns, opt_returns : stitched out-of-sample daily returns
        chosen_params : list of (theta_in, theta_out) per window
            (None entries = fallback to defaults, min-trades unmet)
        last_grid : grid_results DataFrame of the final training window
            (for the sensitivity surface plot)
    """
    n = len(prices)
    results = []

    for pair in pairs:
        ticker_a = pair["ticker_a"]
        ticker_b = pair["ticker_b"]

        default_parts = []
        opt_parts = []
        chosen_params = []
        last_grid = None
        n_windows = 0

        # Same rolling grid as rolling_backtest, skipping steps without
        # enough history for formation + training before the test window
        t_start = formation_period + training_period
        # align to the rolling_backtest grid (multiples of trading_period
        # past the first formation window)
        offset = (t_start - formation_period) % trading_period
        if offset:
            t_start += trading_period - offset

        pair_data = prices[[ticker_a, ticker_b]]

        while t_start + trading_period <= n:
            t_end = t_start + trading_period
            training_start = t_start - training_period
            train_formation_start = training_start - formation_period

            # Skip windows where either ETF lacks data (e.g. not yet launched)
            window_data = pair_data.iloc[train_formation_start:t_end]
            if window_data.notna().mean().min() < 0.9:
                t_start += trading_period
                continue

            try:
                opt = optimize_pair_thresholds(
                    prices, ticker_a, ticker_b,
                    train_formation_start, training_start,
                    training_start, t_start,
                    rolling_window, transaction_cost,
                    theta_in_range, theta_out_range,
                    min_trades=min_trades,
                    stop_loss=stop_loss, max_holding=max_holding,
                    target_vol=target_vol, vol_window=vol_window,
                    max_leverage=max_leverage,
                )
                last_grid = opt["grid_results"]

                if opt["best_theta_in"] is not None:
                    theta_in, theta_out = opt["best_theta_in"], opt["best_theta_out"]
                    chosen_params.append((round(theta_in, 2), round(theta_out, 2)))
                else:
                    theta_in, theta_out = default_theta_in, default_theta_out
                    chosen_params.append(None)

                common_kwargs = dict(
                    rolling_window=rolling_window,
                    transaction_cost=transaction_cost,
                    stop_loss=stop_loss, max_holding=max_holding,
                    target_vol=target_vol, vol_window=vol_window,
                    max_leverage=max_leverage,
                )
                default_result = backtest_pair(
                    prices, ticker_a, ticker_b,
                    t_start - formation_period, t_start, t_start, t_end,
                    default_theta_in, default_theta_out, **common_kwargs,
                )
                opt_result = backtest_pair(
                    prices, ticker_a, ticker_b,
                    t_start - formation_period, t_start, t_start, t_end,
                    theta_in, theta_out, **common_kwargs,
                )

                default_parts.append(default_result["trade_results"]["strategy_return"])
                opt_parts.append(opt_result["trade_results"]["strategy_return"])
                n_windows += 1
            except Exception as e:
                print(f"Walk-forward failed for {ticker_a}-{ticker_b} "
                      f"at window {t_start}: {e}")

            t_start += trading_period

        if n_windows == 0:
            continue

        results.append({
            "ticker_a": ticker_a,
            "ticker_b": ticker_b,
            "n_windows": n_windows,
            "default_returns": pd.concat(default_parts).sort_index(),
            "opt_returns": pd.concat(opt_parts).sort_index(),
            "chosen_params": chosen_params,
            "last_grid": last_grid,
        })

        print(f"{ticker_a}-{ticker_b}: walk-forward over {n_windows} windows done")

    return results


def summarize_walk_forward(results: List[dict]) -> pd.DataFrame:
    """
    Summarize walk-forward results: out-of-sample default vs optimized
    performance, plus parameter-stability diagnostics.

    param_stability is the fraction of optimized windows choosing the
    modal (theta_in, theta_out). Low stability or boundary-hugging modal
    values indicate the optimizer is fitting noise.
    """
    rows = []
    for r in results:
        default_metrics = compute_performance_metrics(r["default_returns"])
        opt_metrics = compute_performance_metrics(r["opt_returns"])

        picked = [p for p in r["chosen_params"] if p is not None]
        if picked:
            counts = pd.Series(picked).value_counts()
            modal_params = counts.index[0]
            stability = counts.iloc[0] / len(picked)
        else:
            modal_params = None
            stability = np.nan

        rows.append({
            "pair": f"{r['ticker_a']}-{r['ticker_b']}",
            "n_windows": r["n_windows"],
            "n_optimized": len(picked),
            "default_sharpe": default_metrics["sharpe_ratio"],
            "opt_sharpe": opt_metrics["sharpe_ratio"],
            "default_return": default_metrics["total_return"],
            "opt_return": opt_metrics["total_return"],
            "modal_params": modal_params,
            "param_stability": stability,
        })

    return pd.DataFrame(rows)
