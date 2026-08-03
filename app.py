import streamlit as st
import plotly.express as px
import pandas as pd

from data_fetcher import fetch_company_data
from monte_carlo import run_monte_carlo_dcf
from backtest import run_holdout_backtest
from database import init_db, save_run, get_history

st.set_page_config(page_title="Probabilistic DCF Engine", layout="wide")
init_db()

st.title("📊 Probabilistic DCF Valuation Engine")
st.caption("Correlated Monte Carlo simulation for intrinsic equity valuation & risk analysis.")

# --- SIDEBAR INPUTS ---
st.sidebar.header("Model Inputs")
ticker_input = st.sidebar.text_input("Ticker Symbol", value="AAPL").strip().upper()
scenario = st.sidebar.selectbox("Macro Scenario", ["Base Case", "High Inflation", "Recession"])
discount_rate = st.sidebar.slider("Base WACC (%)", min_value=5.0, max_value=15.0, value=9.0, step=0.5) / 100.0

run_button = st.sidebar.button("Run Stochastic DCF", type="primary")

# Run simulation when button clicked OR on first load
if run_button or "data" not in st.session_state:
    with st.spinner("Fetching financial statements & running 10,000 simulations..."):
        try:
            data = fetch_company_data(ticker_input)
            st.session_state["data"] = data
            if not data["unsupported"]:
                sim_res = run_monte_carlo_dcf(data, scenario=scenario, wacc=discount_rate)
                st.session_state["sim_res"] = sim_res
                save_run(
                    ticker_input, data["current_price"], sim_res["median"],
                    sim_res["p10"], sim_res["p90"], scenario
                )
        except Exception as e:
            st.error(f"Error executing valuation: {str(e)}")

# --- MAIN DASHBOARD TABS ---
tab1, tab2, tab3 = st.tabs(["🚀 Valuation Engine", "📜 Financial History", "📈 Valuation Tracker"])

if "data" in st.session_state and "sim_res" in st.session_state:
    data = st.session_state["data"]
    
    if data.get("unsupported", False):
        st.warning(f"⚠️ {data['message']}")
    else:
        sim = st.session_state["sim_res"]
        
        # TAB 1: VALUATION ENGINE
        with tab1:
            st.subheader(f"{data['company_name']} ({data['ticker']}) - Intrinsic Value Distribution")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Current Market Price", f"${data['current_price']:.2f}")
            c2.metric("Simulated Median Value", f"${sim['median']:.2f}", 
                      delta=f"{((sim['median'] - data['current_price'])/data['current_price'])*100:.1f}% vs Market")
            c3.metric("10th Percentile (Bear Case)", f"${sim['p10']:.2f}")
            c4.metric("90th Percentile (Bull Case)", f"${sim['p90']:.2f}")
            
            st.info(f"📍 **Market Price Percentile:** Current price sits at the **{sim['price_percentile']:.1f}th percentile** of simulated fair values.")
            
            fig = px.histogram(
                sim["sim_prices"], nbins=60, 
                title="10,000 Simulated Fair Value Prices ($)",
                labels={"value": "Intrinsic Share Price ($)"},
                color_discrete_sequence=["#4F46E5"]
            )
            fig.add_vline(x=data["current_price"], line_dash="dash", line_color="red", annotation_text="Current Price")
            fig.add_vline(x=sim["median"], line_dash="solid", line_color="green", annotation_text="Median Fair Value")
            fig.update_layout(showlegend=False, yaxis_title="Frequency")
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Sensitivity Analysis (Tornado Chart)")
            tornado_df = pd.DataFrame(list(sim["tornado"].items()), columns=["Driver", "Impact Price ($)"])
            fig_torn = px.bar(
                tornado_df, x="Impact Price ($)", y="Driver", orientation="h",
                title="Impact on Fair Value when Drivers Increase (+1σ)",
                color="Impact Price ($)", color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig_torn, use_container_width=True)
            
            st.subheader("🧪 1-Year Holdout Backtest Validation")
            bt = run_holdout_backtest(data, wacc=discount_rate)
            if bt["eligible"]:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.write(f"**Training Window:** {bt['train_years']}")
                    st.write(f"**Tested Holdout Year:** {bt['tested_year']}")
                    st.write(f"**Actual Revenue:** ${bt['actual_rev']/1e9:.2f}B")
                with col_b:
                    st.write(f"**Model Predicted Range (10th-90th):** ${bt['predicted_p10_rev']/1e9:.2f}B - ${bt['predicted_p90_rev']/1e9:.2f}B")
                    if bt["validated"]:
                        st.success("✅ **Validation PASSED:** Actual performance fell within the predicted 80% confidence interval.")
                    else:
                        st.error("❌ **Validation FAILED:** Actual revenue fell outside predicted range.")
            else:
                st.caption("Insufficient historical years to perform holdout validation.")

            with st.expander("📚 Methodology & Limitations"):
                st.markdown("""
                - **Randomization & Correlation:** Uses Cholesky Decomposition to ensure revenue growth, EBIT margins, and exit multiples move together realistically.
                - **Fixed Inputs:** WACC is held fixed for clear traceability.
                - **Data Constraints:** Uses annual financial statements to maximize historical reliability.
                """)

        # TAB 2: FINANCIAL HISTORY
        with tab2:
            st.subheader(f"Historical Statement Metrics ({data['ticker']})")
            st.dataframe(data["financials_df"], use_container_width=True)

        # TAB 3: VALUATION TRACKER
        with tab3:
            st.subheader("Saved Valuation History")
            history_df = get_history(data["ticker"])
            if not history_df.empty:
                st.dataframe(history_df, use_container_width=True)
                fig_hist = px.line(
                    history_df, x="timestamp", y=["median_val", "current_price"],
                    labels={"value": "Price ($)", "timestamp": "Run Date"},
                    title=f"Valuation History Trend: {data['ticker']}"
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.info("No prior valuation runs recorded in local database.")
