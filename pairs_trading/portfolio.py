"""
Portfolio-level aggregation of per-pair strategy returns.

Combines the per-pair daily return streams from the rolling backtest
into one portfolio series (the number that actually matters), with
deduplication of near-identical pairs so the portfolio is not just the
same S&P 500 spread counted three times.
"""

import pandas as pd
import numpy as np
from typing import List

from .backtest import stitch_pair_returns, compute_performance_metrics


# ETFs tracking (near-)identical indices: any two pairs drawing on the
# same group are the same trade and must not both enter the portfolio.
EQUIVALENT_GROUPS = [
    {"SPY", "IVV", "VOO"},   # S&P 500
    {"MDY", "IJH"},          # S&P MidCap 400
    {"AGG", "BND"},          # US aggregate bond
    {"EFA", "VEA"},          # developed ex-US
]


def _canonical(ticker: str) -> str:
    for group in EQUIVALENT_GROUPS:
        if ticker in group:
            return min(group)  # stable group id
    return ticker


def deduplicate_pairs(all_results: List[dict]) -> List[dict]:
    """
    Greedily keep at most one pair per underlying exposure.

    Pairs are ranked by how many windows they were selected in — a
    formation-data quantity, NOT test performance, so the dedup does not
    reintroduce in-sample selection bias. Once a (canonical) ETF is used
    by a kept pair, later pairs touching it are dropped, which also caps
    exposure per underlying index.
    """
    ranked = sorted(
        all_results,
        key=lambda r: (-len(r["windows"]), r["ticker_a"], r["ticker_b"]),
    )
    used = set()
    kept = []
    for r in ranked:
        ca, cb = _canonical(r["ticker_a"]), _canonical(r["ticker_b"])
        if ca in used or cb in used or ca == cb:
            continue
        kept.append(r)
        used.update({ca, cb})
    return kept


def build_portfolio(all_results: List[dict], dedup: bool = True) -> dict:
    """
    Combine per-pair stitched daily returns into a 1/N portfolio.

    Capital is split equally across the N kept pairs; a pair earns 0 on
    days it holds no position or was not selected (idle capital). With
    vol-targeted sizing upstream, equal capital weights are approximately
    equal risk weights.

    Parameters
    ----------
    all_results : list of dict
        The 'results' list from rolling_backtest.
    dedup : bool
        Drop pairs duplicating an underlying exposure first.

    Returns
    -------
    dict with keys:
        portfolio_returns : pd.Series — daily portfolio returns
        pair_returns : pd.DataFrame — per-pair daily returns (NaN when
            the pair was not deployed)
        kept_pairs : list of str
        dropped_pairs : list of str
    """
    kept = deduplicate_pairs(all_results) if dedup else list(all_results)
    kept_keys = {f"{r['ticker_a']}-{r['ticker_b']}" for r in kept}
    dropped = [
        f"{r['ticker_a']}-{r['ticker_b']}"
        for r in all_results
        if f"{r['ticker_a']}-{r['ticker_b']}" not in kept_keys
    ]

    series = {
        f"{r['ticker_a']}-{r['ticker_b']}": stitch_pair_returns(r)
        for r in kept
    }
    pair_returns = pd.DataFrame(series).sort_index()

    n_pairs = pair_returns.shape[1]
    if n_pairs == 0:
        return {
            "portfolio_returns": pd.Series(dtype=float),
            "pair_returns": pair_returns,
            "kept_pairs": [],
            "dropped_pairs": dropped,
        }

    # 1/N allocation; undeployed pairs contribute 0 (idle capital)
    portfolio_returns = pair_returns.fillna(0).sum(axis=1) / n_pairs

    return {
        "portfolio_returns": portfolio_returns,
        "pair_returns": pair_returns,
        "kept_pairs": sorted(kept_keys),
        "dropped_pairs": sorted(dropped),
    }


def report_portfolio(
    portfolio_returns: pd.Series,
    holdout_start: str = None,
) -> pd.DataFrame:
    """
    Portfolio metrics, optionally split into development vs hold-out
    periods. The hold-out row is only meaningful if it was not iterated
    on during development.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Daily portfolio returns.
    holdout_start : str, optional
        Date (YYYY-MM-DD); returns from this date on are reported as the
        hold-out period.

    Returns
    -------
    pd.DataFrame indexed by period name.
    """
    periods = {"full_sample": portfolio_returns}
    if holdout_start is not None:
        cutoff = pd.Timestamp(holdout_start)
        periods["development"] = portfolio_returns[portfolio_returns.index < cutoff]
        periods["holdout"] = portfolio_returns[portfolio_returns.index >= cutoff]

    rows = {}
    for name, rets in periods.items():
        m = compute_performance_metrics(rets)
        rows[name] = {
            "total_return": m["total_return"],
            "ann_return": m["annualized_return"],
            "ann_vol": m["annualized_volatility"],
            "sharpe": m["sharpe_ratio"],
            "t_stat": m["t_stat"],
            "max_drawdown": m["max_drawdown"],
            "n_days": m["num_trading_days"],
        }

    return pd.DataFrame(rows).T
