import yfinance as yf
import pandas as pd
import numpy as np
import requests

UNSUPPORTED_SECTORS = ["Financial Services", "Real Estate"]

CURRENCY_SYMBOLS = {
    "USD": "$", "INR": "\u20b9", "GBP": "\u00a3", "EUR": "\u20ac",
    "JPY": "\u00a5", "CNY": "\u00a5", "AUD": "A$", "CAD": "C$",
    "HKD": "HK$", "SGD": "S$", "CHF": "CHF ",
}


def _historical_tax_rate(financials, default=0.25):
    """Effective tax rate = Tax Provision / Pretax Income, averaged over
    the years available. Falls back to `default` if those line items
    aren't present or there's too little data to trust the average."""
    try:
        tax = financials.loc["Tax Provision"]
        pretax = financials.loc["Pretax Income"]
        rate = (tax / pretax).dropna()
        rate = rate[(rate > 0) & (rate < 0.45)]  # drop one-off/negative-income noise
        if len(rate) >= 2:
            return float(rate.mean())
    except (KeyError, ZeroDivisionError):
        pass
    return default


def fetch_company_data(ticker_symbol):
    ticker_symbol = ticker_symbol.strip().upper()

    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                      '(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
    })

    ticker = yf.Ticker(ticker_symbol, session=session)
    info = ticker.info or {}

    sector = info.get("sector", "Unknown")
    if sector in UNSUPPORTED_SECTORS:
        return {
            "unsupported": True,
            "sector": sector,
            "message": (
                f"DCF models like this one don't work well for {sector.lower()} companies — "
                f"their financials are built around loans, deposits, or property holdings rather "
                f"than typical revenue and costs, so they need a different valuation approach."
            ),
        }

    # Fetch all three statements once, here, rather than re-fetching per tab.
    financials = ticker.financials
    balance_sheet = ticker.balance_sheet
    cashflow = ticker.cashflow

    if financials is None or financials.empty or balance_sheet is None or balance_sheet.empty:
        raise ValueError(
            f"Couldn't retrieve financial statements for '{ticker_symbol}'. Double-check the "
            f"ticker symbol, or try again shortly — Yahoo Finance occasionally rate-limits requests."
        )

    # Chronological order (oldest -> newest) for growth/volatility math.
    financials = financials.iloc[:, ::-1]
    balance_sheet = balance_sheet.iloc[:, ::-1]
    if cashflow is not None and not cashflow.empty:
        cashflow = cashflow.iloc[:, ::-1]

    if financials.shape[1] < 2:
        raise ValueError(
            f"'{ticker_symbol}' only has {financials.shape[1]} year(s) of statements on file — "
            f"not enough history to estimate volatility from. Try a longer-listed company."
        )

    rev = financials.loc["Total Revenue"] if "Total Revenue" in financials.index else financials.loc["Revenue"]
    ebit = financials.loc["EBIT"] if "EBIT" in financials.index else financials.loc["Operating Income"]

    rev_growth = rev.pct_change().dropna()
    margins = (ebit / rev).dropna()

    rev_mean = float(rev_growth.mean()) if len(rev_growth) > 0 else 0.05
    rev_std = float(rev_growth.std()) if len(rev_growth) > 1 and not np.isnan(rev_growth.std()) else 0.03

    margin_mean = float(margins.mean()) if len(margins) > 0 else 0.15
    margin_std = float(margins.std()) if len(margins) > 1 and not np.isnan(margins.std()) else 0.02

    tax_rate = _historical_tax_rate(financials)

    current_price = info.get("currentPrice") or info.get("regularMarketPrice")
    shares_out = info.get("sharesOutstanding")
    market_cap = info.get("marketCap")
    currency = info.get("currency", "USD")
    currency_symbol = CURRENCY_SYMBOLS.get(currency, f"{currency} ")

    if not current_price or not shares_out:
        raise ValueError(
            f"Yahoo Finance didn't return a live price or share count for '{ticker_symbol}'. "
            f"This can happen for delisted or thinly-traded tickers — try a more widely-held stock."
        )

    total_debt = float(balance_sheet.loc["Total Debt"].iloc[-1]) if "Total Debt" in balance_sheet.index else 0.0
    cash = float(balance_sheet.loc["Cash And Cash Equivalents"].iloc[-1]) if "Cash And Cash Equivalents" in balance_sheet.index else 0.0
    net_debt = total_debt - cash

    # Exit multiple derived as EV/EBIT (not EV/EBITDA) so it's unit-consistent with
    # the EBIT-based FCF the simulation actually produces.
    latest_ebit = float(ebit.iloc[-1])
    if market_cap and latest_ebit > 0:
        exit_multiple = (market_cap + net_debt) / latest_ebit
    else:
        exit_multiple = 10.0

    return {
        "unsupported": False,
        "ticker": ticker_symbol,
        "company_name": info.get("shortName", ticker_symbol),
        "sector": sector,
        "currency": currency,
        "currency_symbol": currency_symbol,
        "current_price": float(current_price),
        "shares_out": float(shares_out),
        "total_debt": total_debt,
        "cash": cash,
        "net_debt": float(net_debt),
        "latest_revenue": float(rev.iloc[-1]),
        "latest_ebit": latest_ebit,
        "rev_mean": rev_mean,
        "rev_std": max(rev_std, 0.01),
        "margin_mean": margin_mean,
        "margin_std": max(margin_std, 0.005),
        "tax_rate": tax_rate,
        "exit_multiple": float(np.clip(exit_multiple, 3.0, 35.0)),
        "financials_df": financials,
        "balance_sheet_df": balance_sheet,
        "cashflow_df": cashflow,
        "history_years": len(rev),
    }
