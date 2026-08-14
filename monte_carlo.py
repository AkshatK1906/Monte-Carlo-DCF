import numpy as np
import pandas as pd


def run_monte_carlo_dcf(data, n_sims=10000, scenario="Base Case", wacc=0.09):
    rev_mean = data["rev_mean"]
    margin_mean = data["margin_mean"]
    rev_std = data["rev_std"]
    margin_std = data["margin_std"]
    tax_rate = data["tax_rate"]

    if scenario == "High Inflation":
        wacc += 0.015
        rev_std *= 1.2
        margin_std *= 1.2
    elif scenario == "Recession":
        rev_mean -= 0.04
        margin_mean -= 0.02
        rev_std *= 1.15

    corr_matrix = np.array([
        [1.0,  0.50, 0.40],
        [0.50, 1.0,  0.35],
        [0.40, 0.35, 1.0 ]
    ])

    L = np.linalg.cholesky(corr_matrix)

    uncorrelated_draws = np.random.normal(0, 1, size=(3, n_sims, 5))
    correlated_draws = np.zeros_like(uncorrelated_draws)

    for yr in range(5):
        correlated_draws[:, :, yr] = L @ uncorrelated_draws[:, :, yr]

    sim_rev_growth = rev_mean + correlated_draws[0, :, :] * rev_std
    sim_margins = margin_mean + correlated_draws[1, :, :] * margin_std
    sim_margins = np.clip(sim_margins, -0.20, 0.65)

    mult_std = data["exit_multiple"] * 0.15
    sim_exit_multiples = data["exit_multiple"] + correlated_draws[2, :, 4] * mult_std
    sim_exit_multiples = np.clip(sim_exit_multiples, 3.0, 35.0)

    current_rev = data["latest_revenue"]
    sim_rev = np.zeros((n_sims, 5))
    sim_fcf = np.zeros((n_sims, 5))

    last_rev = np.full(n_sims, current_rev)

    for yr in range(5):
        last_rev = last_rev * (1 + sim_rev_growth[:, yr])
        sim_rev[:, yr] = last_rev
        ebit = sim_rev[:, yr] * sim_margins[:, yr]
        fcf = ebit * (1 - tax_rate)
        sim_fcf[:, yr] = fcf

    discount_factors = (1 + wacc) ** np.arange(1, 6)
    pv_fcf = np.sum(sim_fcf / discount_factors, axis=1)

    # sim_fcf/(1-tax) recovers year-5 EBIT; exit_multiple is EV/EBIT-consistent
    # (derived that way in data_fetcher), so this multiply is unit-correct.
    terminal_ebit = sim_fcf[:, 4] / (1 - tax_rate)
    terminal_value = terminal_ebit * sim_exit_multiples
    pv_terminal = terminal_value / ((1 + wacc) ** 5)

    sim_ev = pv_fcf + pv_terminal
    sim_equity_val = sim_ev - data["net_debt"]
    sim_share_price = sim_equity_val / data["shares_out"]
    sim_share_price = np.maximum(sim_share_price, 0.01)

    p10 = np.percentile(sim_share_price, 10)
    median = np.median(sim_share_price)
    p90 = np.percentile(sim_share_price, 90)

    cur_price = data["current_price"]
    price_percentile = (sim_share_price < cur_price).mean() * 100

    tornado_data = {
        "Revenue growth": np.median(sim_share_price[sim_rev_growth[:, 0] > rev_mean]),
        "Profit margin": np.median(sim_share_price[sim_margins[:, 0] > margin_mean]),
        "Exit multiple": np.median(sim_share_price[sim_exit_multiples > data["exit_multiple"]]),
    }

    return {
        "sim_prices": sim_share_price,
        "median": float(median),
        "p10": float(p10),
        "p90": float(p90),
        "price_percentile": float(price_percentile),
        "wacc_used": wacc,
        "tax_rate_used": tax_rate,
        "tornado": tornado_data,
    }
