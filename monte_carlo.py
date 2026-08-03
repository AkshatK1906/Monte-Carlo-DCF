import numpy as np
import pandas as pd

def run_monte_carlo_dcf(data, n_sims=10000, scenario="Base Case", wacc=0.09, tax_rate=0.25):
    """
    Runs 10,000 correlated Monte Carlo 5-year FCF forecasts using Cholesky Decomposition.
    Correlates Revenue Growth, EBIT Margin, and Terminal Exit Multiple.
    """
    # 1. Apply Macro Scenario Adjustments
    rev_mean = data["rev_mean"]
    margin_mean = data["margin_mean"]
    rev_std = data["rev_std"]
    margin_std = data["margin_std"]
    
    if scenario == "High Inflation":
        wacc += 0.015
        rev_std *= 1.2
        margin_std *= 1.2
    elif scenario == "Recession":
        rev_mean -= 0.04
        margin_mean -= 0.02
        
    # 2. Setup Correlation Matrix (Revenue Growth, Margin, Exit Multiple)
    corr_matrix = np.array([
        [1.0,  0.50, 0.40],
        [0.50, 1.0,  0.35],
        [0.40, 0.35, 1.0 ]
    ])
    
    # Cholesky Decomposition
    L = np.linalg.cholesky(corr_matrix)
    
    # 3. Generate Correlated Random Variables for 5 Forecast Years
    # Shape: (3 variables, n_sims, 5 years)
    uncorrelated_draws = np.random.normal(0, 1, size=(3, n_sims, 5))
    correlated_draws = np.zeros_like(uncorrelated_draws)
    
    for yr in range(5):
        correlated_draws[:, :, yr] = L @ uncorrelated_draws[:, :, yr]
        
    # Unpack Correlated Draws
    sim_rev_growth = rev_mean + correlated_draws[0, :, :] * rev_std
    sim_margins = margin_mean + correlated_draws[1, :, :] * margin_std
    
    # Clip Margins to sane bounds [-20%, 65%]
    sim_margins = np.clip(sim_margins, -0.20, 0.65)
    
    # Exit Multiples (Year 5 draw, correlated with growth)
    mult_std = data["peer_multiple"] * 0.15
    sim_exit_multiples = data["peer_multiple"] + correlated_draws[2, :, 4] * mult_std
    sim_exit_multiples = np.clip(sim_exit_multiples, 3.0, 35.0)
    
    # 4. Vectorized 5-Year DCF Simulation
    current_rev = data["latest_revenue"]
    sim_rev = np.zeros((n_sims, 5))
    sim_fcf = np.zeros((n_sims, 5))
    
    last_rev = np.full(n_sims, current_rev)
    
    for yr in range(5):
        last_rev = last_rev * (1 + sim_rev_growth[:, yr])
        sim_rev[:, yr] = last_rev
        ebit = sim_rev[:, yr] * sim_margins[:, yr]
        fcf = ebit * (1 - tax_rate)  # Simplified NOPAT proxy
        sim_fcf[:, yr] = fcf
        
    # Discount Factors
    discount_factors = (1 + wacc) ** np.arange(1, 6)
    pv_fcf = np.sum(sim_fcf / discount_factors, axis=1)
    
    # Terminal Value (Exit Multiple approach on Year 5 EBITDA/EBIT)
    terminal_value = (sim_fcf[:, 4] / (1 - tax_rate)) * sim_exit_multiples
    pv_terminal = terminal_value / ((1 + wacc) ** 5)
    
    # Enterprise Value & Equity Value per Share
    sim_ev = pv_fcf + pv_terminal
    sim_equity_val = sim_ev - data["net_debt"]
    sim_share_price = sim_equity_val / data["shares_out"]
    
    # Remove negative non-sensical outliers
    sim_share_price = np.maximum(sim_share_price, 0.01)
    
    # Percentile Statistics
    p10 = np.percentile(sim_share_price, 10)
    median = np.median(sim_share_price)
    p90 = np.percentile(sim_share_price, 90)
    
    # Current Price Percentile Rank
    cur_price = data["current_price"]
    price_percentile = (sim_share_price < cur_price).mean() * 100
    
    # 5. Tornado Sensitivity Analysis (Elasticity of Output)
    tornado_data = {
        "Revenue Growth (+1σ)": np.median(sim_share_price[sim_rev_growth[:, 0] > rev_mean]),
        "EBIT Margin (+1σ)": np.median(sim_share_price[sim_margins[:, 0] > margin_mean]),
        "Exit Multiple (+1σ)": np.median(sim_share_price[sim_exit_multiples > data["peer_multiple"]]),
    }
    
    return {
        "sim_prices": sim_share_price,
        "median": float(median),
        "p10": float(p10),
        "p90": float(p90),
        "price_percentile": float(price_percentile),
        "wacc_used": wacc,
        "tornado": tornado_data
    }
