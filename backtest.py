import numpy as np
import pandas as pd

def run_holdout_backtest(data, wacc=0.09):
    """Validates model accuracy using historical holdout testing."""
    financials = data["financials_df"]
    if financials.shape[1] < 3:
        return {"eligible": False, "reason": "Requires at least 3 years of historical statements."}
    
    # Train on earlier years, test on the most recent year
    train_fin = financials.iloc[:, :-1]
    actual_last_yr = financials.iloc[:, -1]
    
    rev_train = train_fin.loc["Total Revenue"] if "Total Revenue" in train_fin.index else train_fin.loc["Revenue"]
    ebit_train = train_fin.loc["EBIT"] if "EBIT" in train_fin.index else train_fin.loc["Operating Income"]
    
    train_growth = rev_train.pct_change().dropna()
    train_margin = (ebit_train / rev_train).dropna()
    
    mean_g = float(train_growth.mean()) if len(train_growth) > 0 else 0.05
    mean_m = float(train_margin.mean()) if len(train_margin) > 0 else 0.15
    
    base_rev = float(rev_train.iloc[-1])
    sim_1yr_rev = base_rev * (1 + np.random.normal(mean_g, max(float(train_growth.std()), 0.02), 5000))
    
    actual_rev = float(actual_last_yr.loc["Total Revenue"] if "Total Revenue" in actual_last_yr.index else actual_last_yr.loc["Revenue"])
    
    p10_rev = float(np.percentile(sim_1yr_rev, 10))
    p90_rev = float(np.percentile(sim_1yr_rev, 90))
    in_range = p10_rev <= actual_rev <= p90_rev
    
    # Safely convert any column header type to a Year string
    start_yr = pd.to_datetime(str(train_fin.columns[0])).strftime('%Y')
    end_yr = pd.to_datetime(str(train_fin.columns[-1])).strftime('%Y')
    test_yr = pd.to_datetime(str(financials.columns[-1])).strftime('%Y')
    
    return {
        "eligible": True,
        "train_years": f"{start_yr} - {end_yr}",
        "tested_year": test_yr,
        "predicted_p10_rev": p10_rev,
        "predicted_p90_rev": p90_rev,
        "actual_rev": actual_rev,
        "validated": in_range
    }
