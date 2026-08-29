"""
Empirical comparison of strategy variants through the honest backtest
(PIT per-window selection, cost filter, stops, vol sizing, forced close).

Every variant is evaluated identically: rolling backtest -> deduplicated
1/N portfolio -> metrics split into development (<2017) and hold-out
(2017+). NOTE: running many variants over the same history is itself
multiple testing — a variant that wins here is a candidate for the full
validation protocol (parameter sensitivity, cost sensitivity, fresh
out-of-sample data), not a proven improvement.
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import pandas as pd
import numpy as np

from pairs_trading.data import get_data, DEFAULT_ETFS
from pairs_trading.backtest import rolling_backtest, summarize_results
from pairs_trading.portfolio import build_portfolio, report_portfolio


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

HOLDOUT_START = "2017-01-01"

# Less-crowded universe: single-country equity + commodity ETFs.
# Cross-border/commodity spreads are less arbitraged than US index
# clones, and pair spreads are wider relative to transaction costs.
ALT_ETFS = [
    "EWA", "EWC", "EWG", "EWJ", "EWU", "EWH", "EWS", "EWT", "EWY", "EWZ",
    "EWL", "EWQ", "EWI", "EWP", "EWD", "EWN", "EWM", "EWK", "EWO", "EZA",
    "GLD", "SLV", "GDX", "USO", "DBC", "DBA",
]

# Shared settings: the improved baseline from the main pipeline.
BASE_PARAMS = dict(
    formation_period=252,
    trading_period=63,
    theta_in=2.0,
    theta_out=1.0,
    rolling_window=200,
    transaction_cost=0.001,
    pit_selection=True,
    correlation_threshold=0.8,
    cointegration_pvalue=0.05,
    min_profit_cost_ratio=3.0,
    stop_loss=3.5,
    max_holding=40,
    target_vol=0.005,
    vol_window=60,
    max_leverage=3.0,
)

# Variant name -> (universe, parameter overrides)
VARIANTS = {
    "baseline (US index ETFs)": ("core", {}),
    "strict selection (p<0.01 + persistence)": (
        "core", dict(cointegration_pvalue=0.01, require_persistence=True)),
    "half-life gate (5-40d)": (
        "core", dict(half_life_range=(5, 40))),
    "strict + half-life": (
        "core", dict(cointegration_pvalue=0.01, require_persistence=True,
                     half_life_range=(5, 40))),
    "alt universe (country/commodity)": ("alt", {}),
    "alt universe + strict + half-life": (
        "alt", dict(cointegration_pvalue=0.01, require_persistence=True,
                    half_life_range=(5, 40))),
}


def run_variant(name: str, prices: pd.DataFrame, overrides: dict) -> dict:
    params = {**BASE_PARAMS, **overrides}
    out = rolling_backtest(prices, candidate_pairs=None, **params)

    row = {
        "variant": name,
        "pairs_traded": len(out["results"]),
        "total_trades": 0,
        "full_sharpe": np.nan, "dev_sharpe": np.nan, "holdout_sharpe": np.nan,
        "full_ann_ret": np.nan, "full_maxdd": np.nan, "full_tstat": np.nan,
    }
    if not out["results"]:
        return row

    summary = summarize_results(out["results"], out["n_windows"])
    row["total_trades"] = int(summary["total_trades"].sum())

    port = build_portfolio(out["results"], dedup=True)
    if len(port["portfolio_returns"]) == 0:
        return row

    report = report_portfolio(port["portfolio_returns"], HOLDOUT_START)
    row["full_sharpe"] = report.loc["full_sample", "sharpe"]
    row["full_ann_ret"] = report.loc["full_sample", "ann_return"]
    row["full_maxdd"] = report.loc["full_sample", "max_drawdown"]
    row["full_tstat"] = report.loc["full_sample", "t_stat"]
    if "development" in report.index:
        row["dev_sharpe"] = report.loc["development", "sharpe"]
    if "holdout" in report.index:
        row["holdout_sharpe"] = report.loc["holdout", "sharpe"]
    row["n_kept_pairs"] = len(port["kept_pairs"])
    return row


def main():
    print("Loading core universe (US index ETFs)...")
    core_prices, _ = get_data(tickers=DEFAULT_ETFS,
                              start_date="2000-01-01", end_date="2024-12-31")
    print("Loading alternative universe (country/commodity ETFs)...")
    alt_prices, _ = get_data(tickers=ALT_ETFS,
                             start_date="2000-01-01", end_date="2024-12-31")
    universes = {"core": core_prices, "alt": alt_prices}

    rows = []
    for name, (universe, overrides) in VARIANTS.items():
        print(f"\n=== {name} ===")
        rows.append(run_variant(name, universes[universe], overrides))

    comparison = pd.DataFrame(rows)
    print("\n" + "=" * 70)
    print("VARIANT COMPARISON (portfolio-level, dev = pre-2017, "
          "holdout = 2017+)")
    print("=" * 70)
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "variant_comparison.csv"
    comparison.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
