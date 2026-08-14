import numpy as np
import pandas as pd


def run_holdout_backtest(data, wacc=0.09):
    """Pretends we only had data through the second-to-last year, forecasts
    revenue one year ahead from there, and checks whether what actually
    happened landed inside that forecast's range."""
    financials = data["financials_df"]

    # Need >=3 training years (>=2 growth observations, so std is a real
    # number and not NaN) plus 1 held-out year to test against.
    if financials.shape[1] < 4:
        return {
            "eligible": False,
            "reason": "needs at least 4 years of statements to hold one out "
                       "and still have enough left to estimate volatility from.",
        }

    train_fin = financials.iloc[:, :-1]
    actual_last_yr = financials.iloc[:, -1]

    rev_train = train_fin.loc["Total Revenue"] if "Total Revenue" in train_fin.index else train_fin.loc["Revenue"]
    train_growth = rev_train.pct_change().dropna()

    mean_g = float(train_growth.mean())
    std_g = float(train_growth.std())
    if np.isnan(std_g):
        std_g = 0.02
    std_g = max(std_g, 0.02)

    base_rev = float(rev_train.iloc[-1])
    sim_1yr_rev = base_rev * (1 + np.random.normal(mean_g, std_g, 5000))

    actual_rev = float(actual_last_yr.loc["Total Revenue"] if "Total Revenue" in actual_last_yr.index else actual_last_yr.loc["Revenue"])

    p10_rev = float(np.percentile(sim_1yr_rev, 10))
    p90_rev = float(np.percentile(sim_1yr_rev, 90))
    in_range = p10_rev <= actual_rev <= p90_rev

    start_yr = pd.to_datetime(str(train_fin.columns[0])).strftime('%Y')
    end_yr = pd.to_datetime(str(train_fin.columns[-1])).strftime('%Y')
    test_yr = pd.to_datetime(str(financials.columns[-1])).strftime('%Y')

    return {
        "eligible": True,
        "train_years": f"{start_yr}\u2013{end_yr}",
        "tested_year": test_yr,
        "predicted_p10_rev": p10_rev,
        "predicted_p90_rev": p90_rev,
        "actual_rev": actual_rev,
        "validated": in_range,
    }
