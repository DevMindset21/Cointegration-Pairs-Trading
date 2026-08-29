"""
Trading signal generation based on z-score thresholds.
"""

import pandas as pd
import numpy as np


def generate_signals(
    zscore: pd.Series,
    theta_in: float = 2.0,
    theta_out: float = 1.0,
    stop_loss: float = None,
    max_holding: int = None,
    force_close_end: bool = True,
) -> pd.DataFrame:
    """
    Generate trading signals from z-score series.

    Signal rules (from the paper):
        - LONG spread when z < -theta_in  (buy A, sell B)
        - SHORT spread when z > theta_in   (sell A, buy B)
        - EXIT when |z| < theta_out

    Risk-management overlays:
        - Stop-loss: exit when |z| > stop_loss (cointegration-breakdown
          protection). Re-entry is blocked until |z| falls back below
          theta_in, so the strategy does not immediately re-enter the
          same diverging trade.
        - Time stop: exit after max_holding days in a position, with the
          same re-entry block.
        - Force close: flatten on the final bar so every window realizes
          its P&L and pays the exit cost.

    Parameters
    ----------
    zscore : pd.Series
        Rolling z-score of the spread.
    theta_in : float
        Entry threshold (absolute z-score to open a position).
    theta_out : float
        Exit threshold (absolute z-score to close a position).
    stop_loss : float, optional
        Absolute z-score beyond which an open position is stopped out.
    max_holding : int, optional
        Maximum number of days to hold a position.
    force_close_end : bool
        Flatten the position on the last bar of the series.

    Returns
    -------
    pd.DataFrame
        Columns: zscore, signal, position
        signal: 1 (long spread), -1 (short spread), 0 (exit/flat)
        position: carried forward position accounting for holds
    """
    signals = pd.DataFrame(index=zscore.index)
    signals["zscore"] = zscore

    # Raw signal based on thresholds
    signals["signal"] = 0
    signals.loc[zscore > theta_in, "signal"] = -1   # short spread
    signals.loc[zscore < -theta_in, "signal"] = 1    # long spread
    signals.loc[zscore.abs() < theta_out, "signal"] = 0  # exit

    # Build position: hold until an exit signal
    position = np.zeros(len(signals))
    zvalues = signals["zscore"].values
    blocked = False  # re-entry blocked after a stop until |z| < theta_in
    holding = 0

    for i in range(1, len(signals)):
        z = zvalues[i]
        prev_pos = position[i - 1]

        if np.isnan(z):
            position[i] = 0
            blocked = False
            holding = 0
            continue

        if prev_pos == 0:
            if blocked:
                if abs(z) < theta_in:
                    blocked = False
                position[i] = 0
            elif z > theta_in:
                position[i] = -1  # short spread
                holding = 1
            elif z < -theta_in:
                position[i] = 1   # long spread
                holding = 1
            else:
                position[i] = 0
        else:
            holding += 1
            if stop_loss is not None and abs(z) > stop_loss:
                position[i] = 0  # stopped out
                blocked = True
                holding = 0
            elif max_holding is not None and holding > max_holding:
                position[i] = 0  # time stop
                blocked = True
                holding = 0
            elif abs(z) < theta_out:
                position[i] = 0  # exit
                holding = 0
            elif prev_pos == 1 and z > theta_in:
                position[i] = -1  # flip to short
                holding = 1
            elif prev_pos == -1 and z < -theta_in:
                position[i] = 1   # flip to long
                holding = 1
            else:
                position[i] = prev_pos  # hold

    if force_close_end and len(position) > 0:
        position[-1] = 0

    signals["position"] = position
    return signals


def compute_trade_returns(
    prices: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    signals: pd.DataFrame,
    beta: float,
    transaction_cost: float = 0.001,
    target_vol: float = None,
    vol_window: int = 60,
    max_leverage: float = 3.0,
) -> pd.DataFrame:
    """
    Compute daily returns from the pairs trading strategy.

    When position = 1 (long spread): long A, short B
    When position = -1 (short spread): short A, long B

    Volatility targeting (optional): each trade is sized at entry to
    target_vol daily volatility, using the rolling std of the normalized
    spread return up to the entry day (no look-ahead). Leverage is locked
    for the life of the trade to avoid daily rebalancing churn, and capped
    at max_leverage. Trades with no vol estimate available are skipped
    (sized at 0).

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted close prices (full history; used for return and vol
        computation — only data up to each date is used at that date).
    ticker_a : str
        First ETF ticker.
    ticker_b : str
        Second ETF ticker.
    signals : pd.DataFrame
        Output from generate_signals (must contain 'position' column).
    beta : float
        Hedge ratio.
    transaction_cost : float
        Cost per unit of position change.
    target_vol : float, optional
        Target daily volatility per pair (e.g. 0.005 = 50 bps/day).
        None disables sizing (position is ±1 unit of capital).
    vol_window : int
        Rolling window (days) for the spread-return vol estimate.
    max_leverage : float
        Cap on the vol-targeting leverage.

    Returns
    -------
    pd.DataFrame
        Columns: position, sized_position, leverage, return_a, return_b,
                 spread_return, strategy_return, cumulative_return, trade_cost
    """
    # Full-history returns so the vol estimate has data preceding the
    # trading window
    ret_a_full = prices[ticker_a].pct_change()
    ret_b_full = prices[ticker_b].pct_change()
    norm_spread_ret_full = (ret_a_full - beta * ret_b_full) / (1 + abs(beta))

    common_idx = prices.index.intersection(signals.index)
    result = pd.DataFrame(index=common_idx)

    result["position"] = signals.loc[common_idx, "position"]
    result["return_a"] = ret_a_full.loc[common_idx]
    result["return_b"] = ret_b_full.loc[common_idx]

    # Strategy return: position * (return_a - beta * return_b)
    # Normalized by (1 + beta) to account for capital allocation
    result["spread_return"] = result["return_a"] - beta * result["return_b"]

    if target_vol is not None:
        vol = norm_spread_ret_full.rolling(
            vol_window, min_periods=vol_window // 2
        ).std()
        leverage_full = (target_vol / vol).clip(upper=max_leverage).fillna(0.0)
        leverage = leverage_full.loc[common_idx]
    else:
        leverage = pd.Series(1.0, index=common_idx)

    # Lock leverage at entry for the life of each trade
    pos = result["position"]
    is_entry = (pos != pos.shift(1)) & (pos != 0)
    locked_leverage = leverage.where(is_entry).ffill().fillna(0.0)
    result["leverage"] = locked_leverage
    result["sized_position"] = pos * locked_leverage

    result["strategy_return"] = (
        result["sized_position"].shift(1) * result["spread_return"] / (1 + abs(beta))
    )

    # Transaction costs on (sized) position changes
    position_change = result["sized_position"].diff().abs()
    result["trade_cost"] = position_change * transaction_cost
    result["strategy_return"] = result["strategy_return"] - result["trade_cost"]

    # Fill NaN
    result["strategy_return"] = result["strategy_return"].fillna(0)

    # Cumulative return
    result["cumulative_return"] = (1 + result["strategy_return"]).cumprod() - 1

    return result
