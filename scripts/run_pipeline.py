"""
Main runner for the Cointegration-Based ETF Pairs Trading strategy.

Orchestrates the full pipeline:
1. Data download and preprocessing
2. Rolling backtest with full PIT methodology (per-window pair
   selection + hedge-ratio estimation, cost filter, risk overlays)
3. Walk-forward parameter optimization (out-of-sample default vs
   optimized comparison)
4. Portfolio aggregation with dedup and development/hold-out split
5. Visualization and reporting
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from pairs_trading.data import get_data, DEFAULT_ETFS
from pairs_trading.spread import compute_pair_spread_and_zscore
from pairs_trading.strategy import generate_signals
from pairs_trading.backtest import rolling_backtest, summarize_results
from pairs_trading.optimize import walk_forward_optimize, summarize_walk_forward
from pairs_trading.portfolio import build_portfolio, report_portfolio
from pairs_trading.visualize import (
    plot_correlation_heatmap,
    plot_pair_prices,
    plot_spread_and_zscore,
    plot_signals_on_zscore,
    plot_portfolio_returns,
    plot_optimization_surface,
    plot_summary_table,
)


# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

CONFIG = {
    # Data
    "tickers": DEFAULT_ETFS,
    "start_date": "2000-01-01",
    "end_date": "2024-12-31",
    "min_history": 252,

    # Pair selection (applied point-in-time, per formation window)
    "correlation_threshold": 0.8,
    "cointegration_pvalue": 0.05,

    # Strategy
    "theta_in": 2.0,
    "theta_out": 1.0,
    "rolling_window": 200,

    # Risk management (fixed a priori — deliberately NOT optimized)
    "stop_loss_z": 3.5,          # z-score stop (cointegration breakdown)
    "max_holding_days": 40,      # time stop
    "target_daily_vol": 0.005,   # ~8% annualized vol target per pair
    "vol_window": 60,
    "max_leverage": 3.0,
    "min_profit_cost_ratio": 3.0,  # expected reversion vs round-trip cost

    # Backtesting
    "formation_period": 252,    # 1 year
    "trading_period": 63,       # 3 months
    "transaction_cost": 0.001,  # 10 bps

    # Walk-forward optimization
    "theta_in_range": (0.5, 3.0, 0.25),
    "theta_out_range": (0.1, 1.5, 0.1),
    "min_trades_opt": 3,

    # Validation
    "holdout_start": "2017-01-01",  # report separately; do not iterate on it

    # How many top pairs to display / optimize in detail
    "top_n_pairs": 5,
}


def main():
    print("=" * 70)
    print("  COINTEGRATION-BASED ETF PAIRS TRADING")
    print("  Based on Chen & Alexiou (2025) — PIT selection + walk-forward")
    print("=" * 70)

    # ── Step 1: Data ──────────────────────────────────────────────────
    print("\n[1/5] Downloading and preprocessing data...")
    prices, log_returns = get_data(
        tickers=CONFIG["tickers"],
        start_date=CONFIG["start_date"],
        end_date=CONFIG["end_date"],
        min_history=CONFIG["min_history"],
    )

    # ── Step 2: Rolling Backtest (PIT selection per window) ──────────
    print("\n[2/5] Running rolling backtest with point-in-time pair selection...")
    backtest_output = rolling_backtest(
        prices,
        candidate_pairs=None,  # full universe; selection is per-window
        formation_period=CONFIG["formation_period"],
        trading_period=CONFIG["trading_period"],
        theta_in=CONFIG["theta_in"],
        theta_out=CONFIG["theta_out"],
        rolling_window=CONFIG["rolling_window"],
        transaction_cost=CONFIG["transaction_cost"],
        pit_selection=True,
        correlation_threshold=CONFIG["correlation_threshold"],
        cointegration_pvalue=CONFIG["cointegration_pvalue"],
        min_profit_cost_ratio=CONFIG["min_profit_cost_ratio"],
        stop_loss=CONFIG["stop_loss_z"],
        max_holding=CONFIG["max_holding_days"],
        target_vol=CONFIG["target_daily_vol"],
        vol_window=CONFIG["vol_window"],
        max_leverage=CONFIG["max_leverage"],
    )
    backtest_results = backtest_output["results"]
    n_windows = backtest_output["n_windows"]

    if not backtest_results:
        print("No pairs were ever selected. Try lowering thresholds.")
        return

    print(f"\nSelection stability (windows selected, of {n_windows} total):")
    print(backtest_output["selection_counts"].head(15).to_string())
    if len(backtest_output["cost_filtered"]) > 0:
        print(f"\nSelected but skipped by cost filter "
              f"({len(backtest_output['cost_filtered'])} pairs, top 15 shown):")
        print(backtest_output["cost_filtered"].head(15).to_string())

    summary = summarize_results(backtest_results, n_windows)
    stable = summary[summary["num_windows"] >= 3]
    print("\nBacktest Summary — stitched daily returns per pair "
          "(sorted by Sharpe; t_stat ~ significance):")
    print(f"(showing {len(stable)} pairs selected in >=3 windows; "
          f"{len(summary) - len(stable)} more traded in fewer windows)")
    print(stable.to_string(index=False))

    # Top pairs by selection frequency (a formation-data quantity, not
    # test performance, to avoid in-sample cherry-picking) — restricted
    # to pairs that actually traded (the most-selected pairs are often
    # index clones removed by the cost filter)
    key_to_result = {
        f"{r['ticker_a']}-{r['ticker_b']}": r for r in backtest_results
    }
    top_keys = [
        k for k in backtest_output["selection_counts"].index
        if k in key_to_result
    ][:CONFIG["top_n_pairs"]]
    top_pairs = [
        {"ticker_a": key_to_result[k]["ticker_a"],
         "ticker_b": key_to_result[k]["ticker_b"]}
        for k in top_keys
    ]

    # ── Step 3: Walk-Forward Optimization ─────────────────────────────
    print(f"\n[3/5] Walk-forward optimization for top {len(top_pairs)} pairs "
          "(train on trailing window, test on next window)...")
    wf_results = walk_forward_optimize(
        prices, top_pairs,
        formation_period=CONFIG["formation_period"],
        training_period=CONFIG["formation_period"],
        trading_period=CONFIG["trading_period"],
        rolling_window=CONFIG["rolling_window"],
        transaction_cost=CONFIG["transaction_cost"],
        theta_in_range=CONFIG["theta_in_range"],
        theta_out_range=CONFIG["theta_out_range"],
        default_theta_in=CONFIG["theta_in"],
        default_theta_out=CONFIG["theta_out"],
        min_trades=CONFIG["min_trades_opt"],
        stop_loss=CONFIG["stop_loss_z"],
        max_holding=CONFIG["max_holding_days"],
        target_vol=CONFIG["target_daily_vol"],
        vol_window=CONFIG["vol_window"],
        max_leverage=CONFIG["max_leverage"],
    )

    wf_summary = pd.DataFrame()
    if wf_results:
        wf_summary = summarize_walk_forward(wf_results)
        print("\nWalk-Forward: Default vs Optimized (all OUT-OF-SAMPLE):")
        print(wf_summary.to_string(index=False))
        print("\nNote: low param_stability or boundary modal_params means "
              "the optimizer is fitting noise — prefer defaults.")

    # ── Step 4: Portfolio Aggregation ─────────────────────────────────
    print("\n[4/5] Building portfolio (1/N across deduplicated pairs)...")
    portfolio = build_portfolio(backtest_results, dedup=True)
    print(f"Kept pairs:    {portfolio['kept_pairs']}")
    print(f"Dropped (duplicate exposure): {portfolio['dropped_pairs']}")

    if len(portfolio["portfolio_returns"]) > 0:
        report = report_portfolio(
            portfolio["portfolio_returns"],
            holdout_start=CONFIG["holdout_start"],
        )
        print(f"\nPortfolio performance (hold-out = {CONFIG['holdout_start']} "
              "onward — do not iterate on it):")
        print(report.to_string())

    # ── Step 5: Visualization ─────────────────────────────────────────
    print("\n[5/5] Generating visualizations...")

    # Correlation heatmap (full sample — diagnostic only)
    plot_correlation_heatmap(log_returns)

    # Detailed plots for top pairs (full-sample spreads, illustrative)
    for p in top_pairs:
        ta, tb = p["ticker_a"], p["ticker_b"]
        plot_pair_prices(prices, ta, tb)

        sd = compute_pair_spread_and_zscore(
            prices, ta, tb, rolling_window=CONFIG["rolling_window"],
        )
        plot_spread_and_zscore(
            sd["spread"], sd["zscore"], ta, tb,
            CONFIG["theta_in"], CONFIG["theta_out"],
        )
        signals = generate_signals(
            sd["zscore"], CONFIG["theta_in"], CONFIG["theta_out"],
            stop_loss=CONFIG["stop_loss_z"],
            max_holding=CONFIG["max_holding_days"],
        )
        plot_signals_on_zscore(
            signals, ta, tb, CONFIG["theta_in"], CONFIG["theta_out"],
        )

    # Portfolio equity curve with hold-out marker
    if len(portfolio["portfolio_returns"]) > 0:
        plot_portfolio_returns(
            portfolio["portfolio_returns"],
            holdout_start=CONFIG["holdout_start"],
        )

    # Sensitivity surface of the last training window for the first pair:
    # a broad plateau around the chosen params = robust; a sharp peak or
    # boundary optimum = overfit
    if wf_results and wf_results[0]["last_grid"] is not None:
        plot_optimization_surface(
            wf_results[0]["last_grid"],
            wf_results[0]["ticker_a"],
            wf_results[0]["ticker_b"],
            value_col="sharpe",
        )

    # Summary table
    if not summary.empty:
        plot_summary_table(summary)

    print("\n" + "=" * 70)
    print("  DONE — All results generated successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
