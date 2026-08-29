# Cointegration-Based ETF Pairs Trading

Implementation of the methodology from:

> **"Cointegration-based pairs trading: identifying and exploiting similar exchange-traded funds"**
> Kezhong Chen, Constantinos Alexiou (Cranfield University, 2025)
> Journal of Asset Management, Vol 26, Issue 5

## Overview

This project implements a cointegration-based pairs trading strategy on ETFs. The pipeline includes:

1. **Data download** — fetches historical ETF prices via yfinance
2. **Point-in-time pair selection** — at every rolling formation window, pairs are re-selected by correlation and Engle-Granger cointegration using only that window's data (no full-sample look-ahead), then screened by a cost filter (expected reversion profit per trade must exceed a multiple of round-trip costs)
3. **Spread & Z-score** — computes spread residuals and rolling z-scores
4. **Trading strategy** — z-score entry/exit signals with a z-score stop-loss, a time stop, volatility-targeted position sizing, and forced close at window end
5. **Backtesting** — rolling formation/trading windows; per-pair daily returns are stitched across windows and Sharpe is computed once on the full series (with a t-stat for significance)
6. **Walk-forward optimization** — thresholds are optimized on a trailing training window (Sharpe objective, minimum-trades constraint) and evaluated on the *following* out-of-sample window; parameter-stability diagnostics flag overfitting
7. **Portfolio aggregation** — 1/N portfolio across pairs after deduplicating near-identical exposures (e.g. SPY/IVV/VOO), reported separately for the development period and a hold-out period
8. **Visualization** — charts for prices, spreads, signals, sensitivity surfaces, and the portfolio equity curve

## Quick Start

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# run from the project root, as modules
python -m scripts.run_pipeline   # PIT backtest, walk-forward, portfolio
python -m analysis.variants      # compare 6 variants across two universes
python -m analysis.validation    # robustness & model-risk suite (see Results)
```

All result tables are written to `results/`.

## Configuration

Key parameters can be adjusted in `scripts/run_pipeline.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `correlation_threshold` | 0.8 | Min correlation for pair filtering (per formation window) |
| `cointegration_pvalue` | 0.05 | Max p-value for cointegration test (per formation window) |
| `theta_in` | 2.0 | Z-score entry threshold |
| `theta_out` | 1.0 | Z-score exit threshold |
| `rolling_window` | 200 | Rolling window for z-score (days) |
| `formation_period` | 252 | Pair selection window (days) |
| `trading_period` | 63 | Trading window (days) |
| `transaction_cost` | 0.001 | Cost per trade (10 bps) |
| `stop_loss_z` | 3.5 | Z-score stop-loss (cointegration-breakdown exit; fixed a priori) |
| `max_holding_days` | 40 | Time stop (fixed a priori) |
| `target_daily_vol` | 0.005 | Per-pair daily vol target for position sizing |
| `max_leverage` | 3.0 | Cap on vol-targeting leverage |
| `min_profit_cost_ratio` | 3.0 | Min expected profit per trade vs round-trip cost |
| `holdout_start` | 2017-01-01 | Hold-out period start (reported separately; do not iterate on it) |

## ETF Universe

SPY, IVV, VOO, QQQ, XLK, XLF, XLE, XLV, XLI, XLP, XLU, XLB, XLY, XLRE, IWM, IWF, IWD, EFA, EEM, VTI, VEA, VWO, AGG, BND, TLT, GLD, SLV, DIA, MDY, IJH

## Project Structure

```
PARIS-TRADING-IMPROVEMENT/
├── README.md
├── requirements.txt
├── pairs_trading/            # core library
│   ├── data.py               # Data download and preprocessing
│   ├── pair_selection.py     # Correlation + cointegration filtering (PIT)
│   ├── spread.py             # Spread, z-score, OU half-life estimation
│   ├── strategy.py           # Signals, stops, vol-targeted sizing
│   ├── backtest.py           # PIT backtest engine, metrics, trade counting
│   ├── optimize.py           # Walk-forward threshold optimization
│   ├── portfolio.py          # Aggregation, exposure dedup, hold-out split
│   └── visualize.py          # Plotting
├── analysis/                 # research studies
│   ├── variants.py           # 6-variant comparison harness
│   └── validation.py         # Robustness & model-risk suite
├── scripts/
│   └── run_pipeline.py       # End-to-end pipeline entry point
└── results/                  # generated tables (CSV)
```

## Results & Validation

**Headline finding: this strategy has no demonstrable edge on liquid US index ETFs at retail transaction costs.** That conclusion is the deliverable — the code exists to establish it rigorously rather than to promote a backtest number.

### Bias correction

The original design selected pairs on the full 2000–2024 sample and then traded them over that same sample, and compared "optimized" against default thresholds on a test window contained inside the optimizer's own training window. Correcting both, plus a Sharpe metric that annualized 63-day returns by compounding, moved the reported portfolio result from +0.43 to **-0.72 Sharpe (t = -2.3)**. Pairs that pass cointegration testing on liquid US index ETFs (IVV-SPY, IJH-MDY, AGG-BND) have spreads too tight to clear 10 bps round-trip costs; the pairs whose spreads do clear costs do not reliably revert.

### Variant study (`variants.py` → `variant_comparison.csv`)

Six configurations were evaluated identically. The best — persistence-filtered selection (p < 0.01, pair must qualify in consecutive windows) on a 26-ETF country/commodity universe — reached **0.89 Sharpe** full-sample and 1.10 on the 2017–2024 hold-out.

### Robustness suite (`validation.py` → `validation_*.csv`)

That 0.89 does **not** survive scrutiny:

| Test | Result | Reading |
|------|--------|---------|
| Deployed exposure | 252 days (6.7% of span) | ~1 year of market exposure across 24 years |
| Newey-West HAC t-stat | 0.99 (p = 0.32) | Not significant |
| Block-bootstrap 95% CI | [-0.90, 2.51] | Straddles zero |
| Deflated Sharpe (N=6 trials) | **0.48** vs 0.94 expected max under null | Fully explained by selection across 6 variants |
| Cost sensitivity | +0.93 @ 5bps → **-1.87 @ 20bps** | Not deployable at realistic costs |
| Regime attribution | +3.26 (2001–08), -2.21 (2009–16) | Crisis-dependent, not persistent |
| Threshold grid (20 combos) | 85% positive, median 0.80 | Genuine plateau — no sharp peak |
| One-at-a-time perturbations (18) | 83% positive; fails without persistence filter | Robust to stops/vol target |

The Deflated Sharpe Ratio is the decisive statistic: with six configurations tried, the expected maximum Sharpe under the null is 0.94, and 0.89 was observed. The correct conclusion is **"no demonstrable edge"** — the sample is too small to prove absence of one.

## References

- Chen, K., & Alexiou, C. (2025). Cointegration-based pairs trading: identifying and exploiting similar exchange-traded funds. *Journal of Asset Management*, 26(5).
- Engle, R. F., & Granger, C. W. J. (1987). Co-integration and error correction: representation, estimation, and testing. *Econometrica*, 55(2), 251–276.
- Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006). Pairs trading: Performance of a relative-value arbitrage rule. *Review of Financial Studies*, 19(3), 797–827.
