import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from data_fetcher import fetch_company_data
from monte_carlo import run_monte_carlo_dcf
from backtest import run_holdout_backtest

st.set_page_config(page_title="Fair Value Finder", layout="wide", initial_sidebar_state="expanded")

# --- COLOR SYSTEM ---
BG = "#0A0C10"
SURFACE = "#13161C"
BORDER = "#242933"
TEXT = "#E9EBEF"
MUTED = "#7C8494"
GREEN = "#00D68F"
RED = "#FF4757"
AMBER = "#FFB020"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
.stApp {{ background-color: {BG}; color: {TEXT}; }}
[data-testid="stSidebar"] {{ background-color: {SURFACE}; border-right: 1px solid {BORDER}; }}
h1, h2, h3, h4 {{ font-family: 'Inter', sans-serif; letter-spacing: -0.01em; color: {TEXT}; }}
[data-testid="stCaptionContainer"] {{ color: {MUTED} !important; }}
[data-testid="stMetric"] {{ background-color: {SURFACE}; border: 1px solid {BORDER}; border-radius: 6px; padding: 14px 16px; }}
[data-testid="stMetricValue"] {{ font-family: 'IBM Plex Mono', monospace; color: {TEXT}; }}
[data-testid="stMetricLabel"] {{ color: {MUTED}; font-size: 12px; letter-spacing: 0.05em; text-transform: uppercase; }}
[data-testid="stMetricDelta"] {{ font-family: 'IBM Plex Mono', monospace; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid {BORDER}; }}
.stTabs [data-baseweb="tab"] {{ background-color: transparent; color: {MUTED}; font-weight: 500; }}
.stTabs [aria-selected="true"] {{ color: {AMBER} !important; border-bottom: 2px solid {AMBER} !important; }}
.stButton button {{ background-color: {AMBER}; color: {BG}; border: none; font-weight: 600; border-radius: 4px; }}
.stButton button:hover {{ background-color: #E6A01C; color: {BG}; }}
[data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 6px; }}
[data-testid="stAlert"] {{ background-color: {SURFACE}; border: 1px solid {BORDER}; color: {TEXT}; }}
[data-testid="stExpander"] {{ background-color: {SURFACE}; border: 1px solid {BORDER}; border-radius: 6px; }}
hr {{ border-color: {BORDER}; }}
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)

st.title("Fair Value Finder")
st.caption("Institutional-grade DCF valuation powered by Monte Carlo simulation.")

# --- SIDEBAR ---
st.sidebar.header("Configuration")
ticker_input = st.sidebar.text_input("Ticker Symbol", value="AAPL").strip().upper()

scenario_map = {
    "Normal Economy (Base Case)": "Base Case",
    "High Inflation & Slower Growth": "High Inflation",
    "Recession (Pessimistic)": "Recession",
}
display_scenario = st.sidebar.selectbox("Economic Scenario", list(scenario_map.keys()))
scenario = scenario_map[display_scenario]

discount_rate = st.sidebar.slider(
    "Required Return / Discount Rate (%)",
    min_value=5.0, max_value=15.0, value=9.0, step=0.5,
    help="Target annual return for taking on this stock's equity risk. Held fixed and "
         "stated explicitly rather than randomized, so it stays a traceable assumption."
) / 100.0

run_button = st.sidebar.button("Run Valuation", type="primary", use_container_width=True)

if "last_ticker" not in st.session_state:
    st.session_state["last_ticker"] = ticker_input

if run_button or "data" not in st.session_state or st.session_state["last_ticker"] != ticker_input:
    st.session_state["last_ticker"] = ticker_input
    st.session_state.pop("data", None)
    st.session_state.pop("sim_res", None)

    if not ticker_input:
        st.sidebar.warning("Enter a ticker symbol.")
    else:
        with st.spinner(f"Fetching financials and running 10,000 simulation paths for {ticker_input}..."):
            try:
                data = fetch_company_data(ticker_input)
                st.session_state["data"] = data
                if not data.get("unsupported", False):
                    sim_res = run_monte_carlo_dcf(data, scenario=scenario, wacc=discount_rate)
                    st.session_state["sim_res"] = sim_res
            except Exception as e:
                st.session_state.pop("data", None)
                st.session_state.pop("sim_res", None)
                st.error(str(e))

# --- MAIN DASHBOARD ---
tab1, tab2 = st.tabs(["Valuation Summary", "DCF Model & Financial Statements"])

if "data" in st.session_state:
    data = st.session_state["data"]
    sym = data.get("currency_symbol", "$")

    if data.get("unsupported", False):
        st.warning(data["message"])
    elif "sim_res" in st.session_state:
        sim = st.session_state["sim_res"]

        # ==========================================
        # TAB 1: VALUATION SUMMARY
        # ==========================================
        with tab1:
            st.subheader(f"{data['company_name']} ({data['ticker']})")

            diff_pct = ((sim['median'] - data['current_price']) / data['current_price']) * 100
            if diff_pct > 10:
                v_label, v_color, v_detail = "UNDERVALUED", GREEN, f"Trading {abs(diff_pct):.1f}% below estimated fair value"
            elif diff_pct < -10:
                v_label, v_color, v_detail = "OVERVALUED", RED, f"Trading {abs(diff_pct):.1f}% above estimated fair value"
            else:
                v_label, v_color, v_detail = "FAIR VALUE", AMBER, f"Within {abs(diff_pct):.1f}% of estimated fair value"

            st.markdown(f"""
            <div style="background:{SURFACE}; border:1px solid {BORDER}; border-left:4px solid {v_color};
                        border-radius:6px; padding:16px 20px; margin-bottom:20px;">
              <span style="font-family:'IBM Plex Mono',monospace; font-size:22px; font-weight:700;
                           color:{v_color}; letter-spacing:0.02em;">{v_label}</span>
              <span style="color:{MUTED}; font-size:14px; margin-left:12px;">{v_detail}</span>
            </div>
            """, unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Market Price", f"{sym}{data['current_price']:,.2f}")
            c2.metric("Fair Value", f"{sym}{sim['median']:,.2f}", delta=f"{diff_pct:.1f}%")
            c3.metric("Worst Case", f"{sym}{sim['p10']:,.2f}", help="10th percentile simulated outcome.")
            c4.metric("Best Case", f"{sym}{sim['p90']:,.2f}", help="90th percentile simulated outcome.")

            # Range bar (signature element)
            p10v, p90v, curv = sim["p10"], sim["p90"], data["current_price"]
            pos_pct = 50.0
            if p90v > p10v:
                pos_pct = max(0.0, min(100.0, (curv - p10v) / (p90v - p10v) * 100))
            st.markdown(f"""
            <div style="margin: 4px 0 24px 0;">
              <div style="display:flex; justify-content:space-between; font-family:'IBM Plex Mono',monospace;
                          font-size:12px; color:{MUTED}; margin-bottom:4px;">
                <span>WORST CASE &middot; {sym}{p10v:,.2f}</span>
                <span>BEST CASE &middot; {sym}{p90v:,.2f}</span>
              </div>
              <div style="position:relative; height:8px; border-radius:4px;
                          background: linear-gradient(90deg, {RED} 0%, {AMBER} 50%, {GREEN} 100%);">
                <div style="position:absolute; left:{pos_pct:.1f}%; top:-6px; width:2px; height:20px;
                            background:#FFFFFF; transform:translateX(-1px);"></div>
              </div>
              <div style="text-align:center; font-family:'IBM Plex Mono',monospace; font-size:12px;
                          color:{TEXT}; margin-top:6px;">
                TODAY'S PRICE &middot; {sym}{curv:,.2f} &middot; {pos_pct:.0f}th percentile of range
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.caption(
                f"Today's price sits at the {sim['price_percentile']:.0f}th percentile of the full "
                f"10,000-simulation distribution — {sim['price_percentile']:.0f}% of simulated outcomes "
                f"came in below today's price."
            )

            # Distribution chart
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=sim["sim_prices"], nbinsx=60,
                marker=dict(color="rgba(233,235,239,0.30)", line=dict(color=BORDER, width=0.5)),
                name="Simulated outcomes",
            ))
            fig.add_vline(x=sim["p10"], line=dict(color=RED, width=2, dash="dot"),
                          annotation_text="Worst Case", annotation_font_color=RED)
            fig.add_vline(x=sim["median"], line=dict(color=AMBER, width=2),
                          annotation_text="Fair Value", annotation_font_color=AMBER)
            fig.add_vline(x=sim["p90"], line=dict(color=GREEN, width=2, dash="dot"),
                          annotation_text="Best Case", annotation_font_color=GREEN)
            fig.add_vline(x=data["current_price"], line=dict(color="#FFFFFF", width=2, dash="dash"),
                          annotation_text="Today", annotation_position="bottom", annotation_font_color="#FFFFFF")
            fig.update_layout(
                template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG,
                font=dict(family="IBM Plex Mono, monospace", color=TEXT),
                title="Distribution of 10,000 Simulated Fair Values",
                xaxis_title=f"Estimated Share Price ({sym.strip()})", yaxis_title="Frequency",
                showlegend=False, margin=dict(t=60, b=40), height=420,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Tornado / sensitivity
            st.markdown("#### What Drives This Valuation")
            st.caption("Median simulated price in scenarios where each driver came in above its average.")
            tornado_df = pd.DataFrame(list(sim["tornado"].items()), columns=["Driver", "Price"]).sort_values("Price")
            fig_torn = go.Figure(go.Bar(
                x=tornado_df["Price"], y=tornado_df["Driver"], orientation="h",
                marker=dict(color=AMBER),
            ))
            fig_torn.update_layout(
                template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG,
                font=dict(family="IBM Plex Mono, monospace", color=TEXT),
                xaxis_title=f"Median Price When Driver Outperforms ({sym.strip()})",
                margin=dict(t=20, b=40), height=260,
            )
            st.plotly_chart(fig_torn, use_container_width=True)

            # Backtest
            st.markdown("#### Historical Accuracy Check")
            bt = run_holdout_backtest(data, wacc=discount_rate)
            if bt["eligible"]:
                st.write(
                    f"Using only data through {bt['train_years']}, the model forecast "
                    f"{bt['tested_year']} revenue. Here's what actually happened:"
                )
                col_a, col_b = st.columns(2)
                col_a.metric(f"Predicted Range ({bt['tested_year']})",
                             f"{sym}{bt['predicted_p10_rev']/1e9:.2f}B \u2013 {sym}{bt['predicted_p90_rev']/1e9:.2f}B")
                col_b.metric(f"Actual ({bt['tested_year']})", f"{sym}{bt['actual_rev']/1e9:.2f}B")
                if bt["validated"]:
                    st.markdown(f'<span style="color:{GREEN}; font-weight:600;">PASSED — actual revenue landed inside the predicted range.</span>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<span style="color:{RED}; font-weight:600;">MISSED — actual revenue fell outside the predicted range.</span>', unsafe_allow_html=True)
            else:
                st.caption(f"Not enough historical data to run this check ({bt.get('reason', '')}).")

            with st.expander("Methodology & limitations"):
                st.markdown(f"""
- **Simulation:** 10,000 correlated 5-year paths for revenue growth, margin, and exit multiple — a weak growth year drags margin down with it, the way it would in reality.
- **Fixed vs. estimated:** the discount rate ({discount_rate*100:.1f}%) is a fixed, stated assumption rather than randomized. The tax rate used ({sim['tax_rate_used']*100:.1f}%) is estimated from the company's own recent filings.
- **Sample size:** built on {data['history_years']} years of annual data — a small sample, so volatility estimates carry real uncertainty.
- **Scope:** not designed for banks, insurers, or REITs. Conglomerates spanning very different business lines are generally better valued in parts (sum-of-the-parts) than as one blended number.
                """)

        # ==========================================
        # TAB 2: DCF MODEL & FINANCIAL STATEMENTS
        # ==========================================
        with tab2:
            st.subheader(f"DCF Valuation Build-Up — {data['company_name']}")
            st.caption("Bridge from enterprise value to per-share intrinsic value.")

            shares = data["shares_out"]
            implied_equity = sim['median'] * shares
            debt = data["total_debt"]
            cash = data["cash"]
            implied_ev = implied_equity + debt - cash

            def fmt(num):
                if num == 0:
                    return "-"
                neg = num < 0
                num = abs(num)
                if num >= 1e9:
                    body = f"{sym}{num/1e9:,.2f}B"
                elif num >= 1e6:
                    body = f"{sym}{num/1e6:,.2f}M"
                else:
                    body = f"{sym}{num:,.0f}"
                return f"({body})" if neg else body

            bridge_rows = [
                ("Assumptions", "Discount Rate (WACC)", f"{discount_rate*100:.1f}%"),
                ("Assumptions", "Economic Scenario", scenario),
                ("Assumptions", "Effective Tax Rate", f"{sim['tax_rate_used']*100:.1f}%"),
                ("Enterprise Value", "Implied Enterprise Value", fmt(implied_ev)),
                ("Equity Bridge", "(-) Total Debt", fmt(-debt)),
                ("Equity Bridge", "(+) Cash & Equivalents", fmt(cash)),
                ("Equity Bridge", "Implied Equity Value", fmt(implied_equity)),
                ("Per Share", "(\u00f7) Shares Outstanding", f"{shares:,.0f}"),
                ("Per Share", "Implied Share Price", f"{sym}{sim['median']:,.2f}"),
            ]
            bridge_df = pd.DataFrame(bridge_rows, columns=["Section", "Line Item", "Value"])
            st.dataframe(bridge_df.set_index(["Section", "Line Item"]), use_container_width=True)

            st.divider()
            st.subheader("Financial Statements")
            st.caption(f"As reported. Figures in millions ({sym.strip()}) unless noted.")

            def prep_statement(df, rows):
                if df is None or df.empty:
                    return pd.DataFrame()
                d = df.copy()
                try:
                    d.columns = pd.to_datetime(d.columns).strftime('%Y')
                except Exception:
                    d.columns = [str(c)[:4] for c in d.columns]
                d = d[sorted(d.columns, reverse=True)[:4]]
                existing = [r for r in rows if r in d.index]
                d = d.loc[existing]
                for col in d.columns:
                    d[col] = pd.to_numeric(d[col], errors='coerce') / 1_000_000
                return d

            def style_statement(df, bold_rows):
                def _bold(row):
                    return ['font-weight:700;' if row.name in bold_rows else '' for _ in row]
                return df.style.apply(_bold, axis=1).format("{:,.1f}", na_rep="-")

            inc_rows = ["Total Revenue", "Cost Of Revenue", "Gross Profit", "Operating Expense",
                        "Operating Income", "EBIT", "EBITDA", "Net Income"]
            inc_df = prep_statement(data["financials_df"], inc_rows)
            if not inc_df.empty:
                st.markdown("**Income Statement**")
                st.dataframe(style_statement(inc_df, {"Total Revenue", "Gross Profit", "Operating Income", "Net Income"}),
                             use_container_width=True)

            cf_rows = ["Operating Cash Flow", "Free Cash Flow", "Capital Expenditure", "Investing Cash Flow", "Financing Cash Flow"]
            cf_df = prep_statement(data.get("cashflow_df"), cf_rows)
            if not cf_df.empty:
                st.markdown("**Cash Flow Statement**")
                st.dataframe(style_statement(cf_df, {"Operating Cash Flow", "Free Cash Flow"}),
                             use_container_width=True)

            st.markdown("**Balance Sheet — Traditional Format**")
            bs = data["balance_sheet_df"]
            assets_df = prep_statement(bs, ["Cash And Cash Equivalents", "Current Assets", "Total Assets"])
            liab_df = prep_statement(bs, ["Current Liabilities", "Total Liabilities Net Minority Interest",
                                           "Total Debt", "Stockholders Equity"])
            bs_col1, bs_col2 = st.columns(2)
            with bs_col1:
                st.caption("ASSETS")
                if not assets_df.empty:
                    st.dataframe(style_statement(assets_df, {"Total Assets"}), use_container_width=True)
            with bs_col2:
                st.caption("LIABILITIES & EQUITY")
                if not liab_df.empty:
                    st.dataframe(style_statement(liab_df, {"Stockholders Equity"}), use_container_width=True)

            try:
                if not bs.empty and "Total Assets" in bs.index and \
                   "Total Liabilities Net Minority Interest" in bs.index and "Stockholders Equity" in bs.index:
                    latest_col = bs.columns[-1]
                    ta = float(bs.loc["Total Assets", latest_col])
                    tl_eq = float(bs.loc["Total Liabilities Net Minority Interest", latest_col]) + \
                            float(bs.loc["Stockholders Equity", latest_col])
                    gap = abs(ta - tl_eq)
                    tol = max(abs(ta) * 0.01, 1)
                    if gap <= tol:
                        st.caption("Balance check: Total Assets = Total Liabilities + Equity (within rounding).")
                    else:
                        st.caption(f"Balance check: a {sym}{gap/1e6:,.1f}M gap between assets and liabilities+equity "
                                   f"(often minority interest or rounding in the source data).")
            except Exception:
                pass
