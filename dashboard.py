import streamlit as st
import pandas as pd
import time

# Page Config
st.set_page_config(
    page_title="AI Traffic Control Dashboard",
    page_icon="🚦",
    layout="wide",
)

# Title
st.title("🚦 AI-Based Adaptive Traffic Signal Control")
st.markdown("### Multi-Agent Reinforcement Learning System")

# Sidebar
st.sidebar.header("System Monitor")
refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 1, 10, 2)
placeholder = st.empty()

# Main Loop to simulate Real-Time updates
while True:
    # 1. Load Data
    try:
        df_stats = pd.read_csv("simulation_stats.csv")
        df_logs = pd.read_csv("decision_log.csv")
    except (FileNotFoundError, pd.errors.EmptyDataError):
        st.error("Waiting for simulation data... Run 'runner.py' first!")
        time.sleep(5)
        continue

    with placeholder.container():
        # KPI Metrics
        kpi1, kpi2, kpi3 = st.columns(3)
        
        # Calculate Metrics
        avg_queue = df_stats["QueueLength"].mean()
        total_fuel = df_stats["FuelConsumption"].sum() / 1000 # Convert to Grams
        emergency_count = df_logs[df_logs["Action"] == "OVERRIDE"].shape[0]

        kpi1.metric(label="🚗 Avg Queue Length", value=f"{avg_queue:.1f} cars")
        kpi2.metric(label="⛽ Total Fuel Consumed", value=f"{total_fuel:.2f} kg")
        kpi3.metric(label="🚑 Emergency Overrides", value=emergency_count)

        st.markdown("---")

        # Charts Row 1
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Live Queue Length (Congestion)")
            # Pivot data to get J1 and J2 as separate lines
            chart_data = df_stats.pivot_table(index="Step", columns="Junction", values="QueueLength")
            st.line_chart(chart_data)

        with col2:
            st.subheader("⚡ Fuel Consumption")
            fuel_data = df_stats.pivot_table(index="Step", columns="Junction", values="FuelConsumption")
            st.line_chart(fuel_data)

        st.markdown("---")

        # Logs Section (XAI)
        st.subheader("🧠 Explainable AI Decision Log")
        
        # Show last 5 decisions
        latest_logs = df_logs.tail(5)[::-1] # Reverse to show newest on top
        
        for index, row in latest_logs.iterrows():
            if row["Action"] == "OVERRIDE":
                st.error(f"**Step {row['Step']}**: {row['Reason']}")
            else:
                st.info(f"**Step {row['Step']}** | **{row['Agent']}** | Action: {row['Action']} | *Reason: {row['Reason']}*")

    # Wait before refreshing
    time.sleep(refresh_rate)