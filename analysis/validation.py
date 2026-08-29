"""
Model-risk and robustness validation for the pairs-trading strategy.

Addresses the four questions a reviewer should ask of any backtested
strategy:

1. Is the point estimate statistically distinguishable from zero once
   autocorrelation and non-normality are accounted for?
   -> Newey-West HAC t-statistics, stationary block-bootstrap CIs.
2. Is it distinguishable from zero once the number of configurations
   tried is accounted for?
   -> Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).
3. Does it survive realistic execution costs?
   -> Cost sensitivity sweep.
4. Does it survive different market conditions and parameter choices?
   -> Regime/subperiod attribution and parameter sensitivity grids.

Run:  python validation.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

from pairs_trading.data import get_data, DEFAULT_ETFS
from pairs_trading.backtest import rolling_backtest, precompute_selections, compute_performance_metrics
from pairs_trading.portfolio import build_portfolio
from analysis.variants import ALT_ETFS, BASE_PARAMS, HOLDOUT_START, RESULTS_DIR


# Winning configuration from the variant study
WINNER_OVERRIDES = dict(
    cointegration_pvalue=0.01,
    require_persistence=True,
    half_life_range=(5, 40),
)

RNG = np.random.default_rng(20240829)


# ──────────────────────────────────────────────────────────────────────
# Statistical inference
# ──────────────────────────────────────────────────────────────────────

def newey_west_tstat(returns: pd.Series, lags: int = None) -> dict:
    """
    HAC (Newey-West) t-statistic for H0: mean daily return = 0.

    Daily strategy returns are autocorrelated (positions are held for
    days to weeks), which makes the naive iid t-statistic anti-
    conservative. Regressing returns on a constant with a HAC covariance
    estimator corrects the standard error for serial correlation and
    heteroskedasticity.

    Parameters
    ----------
    returns : pd.Series
        Daily strategy returns.
    lags : int, optional
        HAC truncation lag. Defaults to the Newey-West rule of thumb
        floor(4 * (T/100)^(2/9)).

    Returns
    -------
    dict with mean, hac_se, t_stat, p_value, lags, n_obs
    """
    r = returns.dropna()
    n = len(r)
    if n < 30:
        return {"mean": np.nan, "hac_se": np.nan, "t_stat": np.nan,
                "p_value": np.nan, "lags": 0, "n_obs": n}

    if lags is None:
        lags = int(np.floor(4 * (n / 100) ** (2 / 9)))

    X = np.ones((n, 1))
    model = OLS(r.values, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})

    return {
        "mean": float(model.params[0]),
        "hac_se": float(model.bse[0]),
        "t_stat": float(model.tvalues[0]),
        "p_value": float(model.pvalues[0]),
        "lags": lags,
        "n_obs": n,
    }


def block_bootstrap_sharpe_ci(
    returns: pd.Series,
    n_boot: int = 5000,
    block: int = 21,
    alpha: float = 0.05,
) -> dict:
    """
    Percentile confidence interval for the annualized Sharpe ratio via a
    circular block bootstrap.

    Resampling in blocks (default one trading month) preserves the serial
    dependence created by multi-day holding periods, which an iid
    bootstrap would destroy — an iid CI on this kind of series is too
    narrow.

    Returns
    -------
    dict with point estimate, ci_low, ci_high, prob_positive
    """
    r = returns.dropna().values
    n = len(r)
    if n < block * 3:
        return {"sharpe": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "prob_positive": np.nan}

    def sharpe(x):
        sd = x.std(ddof=1)
        return x.mean() / sd * np.sqrt(252) if sd > 0 else 0.0

    n_blocks = int(np.ceil(n / block))
    starts = RNG.integers(0, n, size=(n_boot, n_blocks))
    offsets = np.arange(block)
    # circular indexing keeps every bootstrap sample the same length
    idx = (starts[:, :, None] + offsets[None, None, :]) % n
    samples = r[idx.reshape(n_boot, -1)[:, :n]]

    sds = samples.std(axis=1, ddof=1)
    boot = np.where(sds > 0, samples.mean(axis=1) / np.where(sds > 0, sds, 1),
                    0.0) * np.sqrt(252)

    return {
        "sharpe": sharpe(r),
        "ci_low": float(np.percentile(boot, 100 * alpha / 2)),
        "ci_high": float(np.percentile(boot, 100 * (1 - alpha / 2))),
        "prob_positive": float((boot > 0).mean()),
    }


def deflated_sharpe_ratio(
    returns: pd.Series,
    trial_sharpes_annual: np.ndarray,
    n_trials: int = None,
) -> dict:
    """
    Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

    Selecting the best of N configurations inflates the winner's Sharpe
    even when every configuration is worthless: the expected maximum
    Sharpe under the null grows with N and with the dispersion of Sharpes
    across trials. The DSR is the probability that the observed Sharpe
    exceeds that selection-adjusted benchmark, also correcting for the
    skewness and kurtosis of the return distribution.

    DSR > 0.95 is the usual bar for calling a result significant after
    accounting for selection.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the SELECTED (best) configuration.
    trial_sharpes_annual : array-like
        Annualized Sharpes of all configurations tried.
    n_trials : int, optional
        Number of independent trials. Defaults to len(trial_sharpes).

    Returns
    -------
    dict with observed_sharpe, expected_max_sharpe_null, dsr, n_trials
    """
    r = returns.dropna().values
    T = len(r)
    if T < 30:
        return {"dsr": np.nan}

    sd = r.std(ddof=1)
    sr_hat = r.mean() / sd if sd > 0 else 0.0  # per-day units
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=False))  # non-excess

    trials_daily = np.asarray(trial_sharpes_annual, dtype=float) / np.sqrt(252)
    var_trials = float(np.var(trials_daily, ddof=1)) if len(trials_daily) > 1 else 0.0
    N = int(n_trials or len(trials_daily))

    gamma = 0.5772156649015329  # Euler-Mascheroni
    if N > 1 and var_trials > 0:
        z1 = stats.norm.ppf(1 - 1 / N)
        z2 = stats.norm.ppf(1 - 1 / (N * np.e))
        sr_null = np.sqrt(var_trials) * ((1 - gamma) * z1 + gamma * z2)
    else:
        sr_null = 0.0

    denom = np.sqrt(max(1e-12, 1 - skew * sr_hat + ((kurt - 1) / 4) * sr_hat ** 2))
    dsr = float(stats.norm.cdf(((sr_hat - sr_null) * np.sqrt(T - 1)) / denom))

    return {
        "observed_sharpe_ann": sr_hat * np.sqrt(252),
        "expected_max_null_ann": sr_null * np.sqrt(252),
        "dsr": dsr,
        "n_trials": N,
        "skew": skew,
        "kurtosis": kurt,
        "n_obs": T,
    }


# ──────────────────────────────────────────────────────────────────────
# Configuration runner
# ──────────────────────────────────────────────────────────────────────

def filter_selection(selections: dict, pvalue: float) -> dict:
    """Tighten a precomputed selection to a stricter cointegration p-value."""
    return {
        k: [p for p in v if p["p_value"] < pvalue]
        for k, v in selections.items()
    }


def run_config(prices: pd.DataFrame, selections: dict, **overrides) -> pd.Series:
    """
    Run one configuration and return the deduplicated 1/N portfolio's
    daily return series (empty Series when nothing traded).
    """
    params = {**BASE_PARAMS, **WINNER_OVERRIDES, **overrides}
    pvalue = params.pop("cointegration_pvalue")
    sel = filter_selection(selections, pvalue)

    out = rolling_backtest(prices, candidate_pairs=None,
                           selection_by_window=sel, **params)
    if not out["results"]:
        return pd.Series(dtype=float)
    return build_portfolio(out["results"], dedup=True)["portfolio_returns"]


def summarize(returns: pd.Series) -> dict:
    """Compact metric bundle for a return series."""
    if len(returns) == 0:
        return {"sharpe": np.nan, "ann_return": np.nan,
                "max_dd": np.nan, "n_days": 0}
    m = compute_performance_metrics(returns)
    return {
        "sharpe": m["sharpe_ratio"],
        "ann_return": m["annualized_return"],
        "max_dd": m["max_drawdown"],
        "n_days": m["num_trading_days"],
    }


# ──────────────────────────────────────────────────────────────────────
# Robustness analyses
# ──────────────────────────────────────────────────────────────────────

def cost_sensitivity(prices, selections, cost_levels=(0.0005, 0.001, 0.002, 0.003)):
    """
    Re-run the strategy across execution-cost assumptions.

    Both the P&L charge and the profit-to-cost pair screen scale with the
    assumption, which is the realistic coupling: at higher costs fewer
    spreads are worth trading at all.
    """
    rows = []
    for c in cost_levels:
        rets = run_config(prices, selections, transaction_cost=c)
        row = {"cost_bps": c * 1e4, **summarize(rets)}
        if len(rets) > 0:
            row["nw_t"] = newey_west_tstat(rets)["t_stat"]
        rows.append(row)
    return pd.DataFrame(rows)


def parameter_sensitivity(prices, selections,
                          theta_ins=(1.5, 1.75, 2.0, 2.25, 2.5),
                          theta_outs=(0.5, 0.75, 1.0, 1.25)):
    """
    Out-of-sample Sharpe across the entry/exit threshold grid.

    A trustworthy parameter choice sits on a broad plateau: neighbours
    should perform similarly. A sharp peak means the backtest is fitting
    noise, which is what the walk-forward study already suggested.
    """
    rows = []
    for t_in in theta_ins:
        for t_out in theta_outs:
            if t_out >= t_in:
                continue
            rets = run_config(prices, selections,
                              theta_in=t_in, theta_out=t_out)
            rows.append({"theta_in": t_in, "theta_out": t_out,
                         **summarize(rets)})
    return pd.DataFrame(rows)


def one_at_a_time_sensitivity(prices, selections):
    """
    Perturb each design choice individually, holding the rest fixed.

    Isolates which single assumption the result depends on — the
    parameters that were fixed a priori (stops, vol target) should not
    dominate, or the strategy is really a stop-loss artifact.
    """
    perturbations = {
        "base": {},
        "coint p<0.005": dict(cointegration_pvalue=0.005),
        "coint p<0.02": dict(cointegration_pvalue=0.02),
        "half-life 3-60d": dict(half_life_range=(3, 60)),
        "half-life 10-30d": dict(half_life_range=(10, 30)),
        "no half-life gate": dict(half_life_range=None),
        "no persistence": dict(require_persistence=False),
        "stop 3.0 sigma": dict(stop_loss=3.0),
        "stop 4.0 sigma": dict(stop_loss=4.0),
        "no stop-loss": dict(stop_loss=None),
        "time stop 20d": dict(max_holding=20),
        "time stop 60d": dict(max_holding=60),
        "vol target 0.3%": dict(target_vol=0.003),
        "vol target 0.8%": dict(target_vol=0.008),
        "no vol targeting": dict(target_vol=None),
        "z-window 120d": dict(rolling_window=120),
        "profit/cost >=2": dict(min_profit_cost_ratio=2.0),
        "profit/cost >=5": dict(min_profit_cost_ratio=5.0),
    }
    rows = []
    for name, over in perturbations.items():
        rets = run_config(prices, selections, **over)
        rows.append({"perturbation": name, **summarize(rets)})
    return pd.DataFrame(rows)


def regime_analysis(returns: pd.Series, spy: pd.Series) -> pd.DataFrame:
    """
    Attribute performance to market regimes and named stress episodes.

    The volatility split uses an EXPANDING median of trailing realized
    vol, so the calm/stressed label at each date uses only information
    available by that date (no look-ahead in the classifier itself).
    """
    realized_vol = spy.pct_change().rolling(21).std() * np.sqrt(252)
    threshold = realized_vol.expanding(min_periods=252).median()
    stressed = (realized_vol > threshold).reindex(returns.index).fillna(False)

    periods = {
        "2001-2008": returns[returns.index < "2009-01-01"],
        "2009-2016": returns[(returns.index >= "2009-01-01") &
                             (returns.index < "2017-01-01")],
        "2017-2024": returns[returns.index >= "2017-01-01"],
        "GFC (2007-07..2009-06)": returns[(returns.index >= "2007-07-01") &
                                          (returns.index <= "2009-06-30")],
        "COVID (2020-02..2020-04)": returns[(returns.index >= "2020-02-01") &
                                            (returns.index <= "2020-04-30")],
        "2022 bear": returns[(returns.index >= "2022-01-01") &
                             (returns.index <= "2022-10-31")],
        "high-vol days": returns[stressed],
        "low-vol days": returns[~stressed],
    }

    rows = []
    for name, r in periods.items():
        row = {"regime": name, **summarize(r)}
        if len(r) >= 30:
            row["nw_t"] = newey_west_tstat(r)["t_stat"]
        rows.append(row)
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    alt_prices, _ = get_data(tickers=ALT_ETFS,
                             start_date="2000-01-01", end_date="2024-12-31")
    spy_prices, _ = get_data(tickers=["SPY"],
                             start_date="2000-01-01", end_date="2024-12-31")
    spy = spy_prices["SPY"]

    print("\nPrecomputing point-in-time pair selections (p<0.05 superset)...")
    selections = precompute_selections(
        alt_prices,
        formation_period=BASE_PARAMS["formation_period"],
        trading_period=BASE_PARAMS["trading_period"],
        correlation_threshold=BASE_PARAMS["correlation_threshold"],
        cointegration_pvalue=0.05,
    )
    print(f"  {len(selections)} windows cached")

    # ── Baseline configuration ───────────────────────────────────────
    print("\n[1/6] Baseline (winning variant)...")
    base_rets = run_config(alt_prices, selections)
    base = summarize(base_rets)
    calendar_days = len(alt_prices.loc[base_rets.index.min():base_rets.index.max()]) \
        if len(base_rets) else 0
    print(f"  Sharpe {base['sharpe']:.3f} | ann ret {base['ann_return']:.4f} "
          f"| maxDD {base['max_dd']:.4f} | {base['n_days']} deployed days "
          f"({100 * base['n_days'] / max(calendar_days, 1):.1f}% of calendar span)")

    # ── Inference ────────────────────────────────────────────────────
    print("\n[2/6] Statistical inference...")
    nw = newey_west_tstat(base_rets)
    boot = block_bootstrap_sharpe_ci(base_rets)
    print(f"  Newey-West t = {nw['t_stat']:.3f} (p = {nw['p_value']:.3f}, "
          f"{nw['lags']} lags, n = {nw['n_obs']})")
    print(f"  Bootstrap 95% CI for Sharpe: "
          f"[{boot['ci_low']:.3f}, {boot['ci_high']:.3f}]  "
          f"P(Sharpe>0) = {boot['prob_positive']:.3f}")

    # ── Deflated Sharpe ──────────────────────────────────────────────
    print("\n[3/6] Multiple-testing correction...")
    try:
        trials = pd.read_csv(
            RESULTS_DIR / "variant_comparison.csv")["full_sharpe"].dropna().values
    except Exception:
        trials = np.array([base["sharpe"]])
    dsr6 = deflated_sharpe_ratio(base_rets, trials)
    dsr20 = deflated_sharpe_ratio(base_rets, trials, n_trials=20)
    print(f"  Trials recorded: {len(trials)}  "
          f"(Sharpe dispersion {np.std(trials, ddof=1):.3f})")
    print(f"  Expected max Sharpe under null (N=6):  "
          f"{dsr6['expected_max_null_ann']:.3f}  ->  DSR = {dsr6['dsr']:.3f}")
    print(f"  Conservative (N=20):                   "
          f"{dsr20['expected_max_null_ann']:.3f}  ->  DSR = {dsr20['dsr']:.3f}")
    print(f"  Return skew {dsr6['skew']:.2f}, kurtosis {dsr6['kurtosis']:.2f}")

    # ── Cost sensitivity ─────────────────────────────────────────────
    print("\n[4/6] Cost sensitivity...")
    costs = cost_sensitivity(alt_prices, selections)
    print(costs.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # ── Regime analysis ──────────────────────────────────────────────
    print("\n[5/6] Regime & subperiod attribution...")
    regimes = regime_analysis(base_rets, spy)
    print(regimes.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # ── Parameter sensitivity ────────────────────────────────────────
    print("\n[6/6] Parameter sensitivity...")
    grid = parameter_sensitivity(alt_prices, selections)
    print("\nThreshold grid (out-of-sample Sharpe):")
    pivot = grid.pivot_table(values="sharpe", index="theta_out", columns="theta_in")
    print(pivot.to_string(float_format=lambda x: f"{x:.3f}"))
    pos = (grid["sharpe"] > 0).mean()
    print(f"\n  Grid points with Sharpe > 0: {pos:.0%} "
          f"({len(grid)} combinations)")
    print(f"  Sharpe range across grid: [{grid['sharpe'].min():.3f}, "
          f"{grid['sharpe'].max():.3f}], median {grid['sharpe'].median():.3f}")

    oat = one_at_a_time_sensitivity(alt_prices, selections)
    print("\nOne-at-a-time perturbations:")
    print(oat.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\n  Perturbations with Sharpe > 0: "
          f"{(oat['sharpe'] > 0).mean():.0%} of {len(oat)}")

    # ── Persist ──────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(exist_ok=True)
    costs.to_csv(RESULTS_DIR / "validation_costs.csv", index=False)
    regimes.to_csv(RESULTS_DIR / "validation_regimes.csv", index=False)
    grid.to_csv(RESULTS_DIR / "validation_param_grid.csv", index=False)
    oat.to_csv(RESULTS_DIR / "validation_perturbations.csv", index=False)
    pd.DataFrame([{
        "sharpe": base["sharpe"], "ann_return": base["ann_return"],
        "max_dd": base["max_dd"], "deployed_days": base["n_days"],
        "nw_t_stat": nw["t_stat"], "nw_p_value": nw["p_value"],
        "boot_ci_low": boot["ci_low"], "boot_ci_high": boot["ci_high"],
        "boot_prob_positive": boot["prob_positive"],
        "dsr_n6": dsr6["dsr"], "dsr_n20": dsr20["dsr"],
        "skew": dsr6["skew"], "kurtosis": dsr6["kurtosis"],
    }]).to_csv(RESULTS_DIR / "validation_inference.csv", index=False)
    print(f"\nSaved to {RESULTS_DIR}/: validation_"
          "{inference,costs,regimes,param_grid,perturbations}.csv")


if __name__ == "__main__":
    main()
