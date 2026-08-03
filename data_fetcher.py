import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st

UNSUPPORTED_SECTORS = ["Financial Services", "Real Estate"]

@st.cache_data(ttl=3600)  # Caches yfinance data for 1 hour to prevent rate limiting!
def fetch_company_data(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)
    info = ticker.info
    
    # 1. Sector Safety Check
    sector = info.get("sector", "Unknown")
    if sector in UNSUPPORTED_SECTORS:
        return {
            "unsupported": True,
            "sector": sector,
            "message": f"DCF models are unsuitable for balance-sheet driven businesses ({sector}). Use a DDM or Residual Income model instead."
        }
    
    # 2. Pull Financial Statements (Annual)
    financials = ticker.financials
    balance_sheet = ticker.balance_sheet
    
    if financials.empty or balance_sheet.empty:
        raise ValueError(f"Could not retrieve complete financial statements for '{ticker_symbol}'. Check ticker symbol.")
    
    financials = financials.iloc[:, ::-1]
    balance_sheet = balance_sheet.iloc[:, ::-1]
    
    rev = financials.loc["Total Revenue"] if "Total Revenue" in financials.index else financials.loc["Revenue"]
    ebit = financials.loc["EBIT"] if "EBIT" in financials.index else financials.loc["Operating Income"]
    
    rev_growth = rev.pct_change().dropna()
    margins = (ebit / rev).dropna()
    
    rev_mean = float(rev_growth.mean()) if len(rev_growth) > 0 else 0.05
    rev_std = float(rev_growth.std()) if len(rev_growth) > 1 and not np.isnan(rev_growth.std()) else 0.03
    
    margin_mean = float(margins.mean()) if len(margins) > 0 else 0.15
    margin_std = float(margins.std()) if len(margins) > 1 and not np.isnan(margins.std()) else 0.02
    
    current_price = info.get("currentPrice") or info.get("regularMarketPrice") or float(rev.iloc[-1] / 100)
    shares_out = info.get("sharesOutstanding") or 1_000_000
    
    total_debt = float(balance_sheet.loc["Total Debt"].iloc[-1]) if "Total Debt" in balance_sheet.index else 0.0
    cash = float(balance_sheet.loc["Cash And Cash Equivalents"].iloc[-1]) if "Cash And Cash Equivalents" in balance_sheet.index else 0.0
    net_debt = total_debt - cash
    
    peer_multiple = info.get("enterpriseToEbitda") or 10.0
    if peer_multiple <= 0 or np.isnan(peer_multiple):
        peer_multiple = 10.0
        
    return {
        "unsupported": False,
        "ticker": ticker_symbol.upper(),
        "company_name": info.get("shortName", ticker_symbol),
        "sector": sector,
        "current_price": float(current_price),
        "shares_out": float(shares_out),
        "net_debt": float(net_debt),
        "latest_revenue": float(rev.iloc[-1]),
        "latest_ebit": float(ebit.iloc[-1]),
        "rev_mean": rev_mean,
        "rev_std": max(rev_std, 0.01),
        "margin_mean": margin_mean,
        "margin_std": max(margin_std, 0.005),
        "peer_multiple": float(peer_multiple),
        "financials_df": financials,
        "history_years": len(rev)
    }
