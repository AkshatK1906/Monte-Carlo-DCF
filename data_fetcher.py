import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import requests

UNSUPPORTED_SECTORS = ["Financial Services", "Real Estate"]

def get_demo_data(ticker_symbol):
    """Fallback data generator when Yahoo Finance rate-limits public cloud IPs."""
    ticker_symbol = ticker_symbol.strip().upper()
    
    # Pre-packaged financial data for popular tickers
    demo_db = {
        "AAPL": {"name": "Apple Inc.", "sector": "Technology", "price": 225.0, "rev": 385e9, "ebit": 115e9, "growth": 0.06, "margin": 0.30, "multiple": 28.0, "debt": 100e9, "cash": 60e9, "shares": 15.3e9},
        "MSFT": {"name": "Microsoft Corp.", "sector": "Technology", "price": 440.0, "rev": 245e9, "ebit": 109e9, "growth": 0.12, "margin": 0.44, "multiple": 32.0, "debt": 75e9, "cash": 80e9, "shares": 7.4e9},
        "NVDA": {"name": "NVIDIA Corp.", "sector": "Technology", "price": 120.0, "rev": 96e9, "ebit": 55e9, "growth": 0.40, "margin": 0.57, "multiple": 45.0, "debt": 10e9, "cash": 31e9, "shares": 24.5e9},
        "GOOGL": {"name": "Alphabet Inc.", "sector": "Technology", "price": 175.0, "rev": 307e9, "ebit": 84e9, "growth": 0.10, "margin": 0.27, "multiple": 22.0, "debt": 28e9, "cash": 110e9, "shares": 12.4e9},
        "TSLA": {"name": "Tesla Inc.", "sector": "Consumer Cyclical", "price": 220.0, "rev": 96e9, "ebit": 9e9, "growth": 0.15, "margin": 0.09, "multiple": 50.0, "debt": 5e9, "cash": 29e9, "shares": 3.2e9}
    }
    
    d = demo_db.get(ticker_symbol, {
        "name": f"{ticker_symbol} Corp (Benchmark)", "sector": "Technology", "price": 150.0, 
        "rev": 50e9, "ebit": 10e9, "growth": 0.08, "margin": 0.20, "multiple": 18.0, 
        "debt": 10e9, "cash": 5e9, "shares": 1e9
    })
    
    # Generate 4 years of synthetic historical statements for Tab 2 & Backtests
    years = ["2021-12-31", "2022-12-31", "2023-12-31", "2024-12-31"]
    rev_history = [d["rev"] * ((1 + d["growth"]) ** (i - 3)) for i in range(4)]
    ebit_history = [r * d["margin"] for r in rev_history]
    
    fin_df = pd.DataFrame({
        y: [r, e] for y, r, e in zip(years, rev_history, ebit_history)
    }, index=["Total Revenue", "EBIT"])
    
    return {
        "unsupported": False,
        "is_fallback": True,
        "ticker": ticker_symbol,
        "company_name": d["name"],
        "sector": d["sector"],
        "current_price": float(d["price"]),
        "shares_out": float(d["shares"]),
        "net_debt": float(d["debt"] - d["cash"]),
        "latest_revenue": float(rev_history[-1]),
        "latest_ebit": float(ebit_history[-1]),
        "rev_mean": float(d["growth"]),
        "rev_std": 0.04,
        "margin_mean": float(d["margin"]),
        "margin_std": 0.02,
        "peer_multiple": float(d["multiple"]),
        "financials_df": fin_df,
        "history_years": 4
    }

@st.cache_data(ttl=3600)
def fetch_company_data(ticker_symbol):
    ticker_symbol = ticker_symbol.strip().upper()
    
    try:
        # 1. Custom Session with Browser Headers to reduce 429 blocks
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        
        ticker = yf.Ticker(ticker_symbol, session=session)
        info = ticker.info
        
        # Sector Check
        sector = info.get("sector", "Unknown")
        if sector in UNSUPPORTED_SECTORS:
            return {
                "unsupported": True,
                "is_fallback": False,
                "sector": sector,
                "message": f"DCF models are unsuitable for balance-sheet driven businesses ({sector}). Use a DDM or Residual Income model instead."
            }
        
        financials = ticker.financials
        balance_sheet = ticker.balance_sheet
        
        if financials is None or financials.empty or balance_sheet is None or balance_sheet.empty:
            raise ValueError("Empty financial data returned from Yahoo Finance.")
        
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
            "is_fallback": False,
            "ticker": ticker_symbol,
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
    except Exception:
        # Fallback to demo mode if Yahoo Finance rate-limits Streamlit's IP
        return get_demo_data(ticker_symbol)
