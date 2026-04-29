
import streamlit as st
import obd
import time
import numpy as np
import joblib
import pandas as pd
from datetime import datetime, timedelta

st.set_page_config(page_title='OilAlert Pro: Precision', layout='wide')
st.title(' OilAlert ')

# Load models
rf_model = joblib.load('oil_degradation_rf_model.pkl')
feature_scaler = joblib.load('feature_scaler.pkl')

if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=['Time', 'MAP', 'RPM', 'Throttle', 'Temp', 'Health', 'Stress_Factor', 'Behavior_Label'])

# --- SIDEBAR: AUTOMATED SPECS ---
st.sidebar.header('📋 Live Vehicle Behavior')
behavior_placeholder = st.sidebar.empty()
behavior_placeholder.info("Awaiting connection...")

# --- MAIN UI ---
st.subheader("Live Sensor Data")
cols = st.columns(4)
m_map, m_rpm, m_thr, m_tmp = cols[0].empty(), cols[1].empty(), cols[2].empty(), cols[3].empty()
# Show initial state
m_map.metric('MAP', "0.0 kPa")
m_rpm.metric('RPM', "0")
m_thr.metric('Throttle', "0.0 %")
m_tmp.metric('Temp', "0.0 °C")

st.divider()

st.subheader("Oil Health Status")
health_bar_placeholder = st.empty()
health_bar_placeholder.progress(0, text="Oil Health: --%")

f_col1, f_col2 = st.columns(2)
with f_col1: forecast_placeholder = st.empty()
with f_col2: st.write("")

st.divider()
st.subheader("5-Minute Interval Averages (Last 3)")
avg_placeholder = st.empty()

chart_placeholder = st.empty()

st.divider()
st.subheader('🌡️ Thermal Stress Indicators')
thermal_placeholder = st.empty()
thermal_placeholder.info("No thermal data yet.")

def process_behavior(rpm):
    if 500 <= rpm <= 1000: return 1.0, "💤 Engine Idling"
    elif 1001 <= rpm <= 2000: return 1.2, "🍃 Normal Driving"
    elif rpm > 2000: return 2.5, "🔥 Aggressive Driving"
    else: return 1.0, "⚪ Engine Off/Low"

def update_ui():
    if not st.session_state.history.empty:
        last = st.session_state.history.iloc[-1]
        m_map.metric('MAP', f"{last['MAP']:.1f} kPa")
        m_rpm.metric('RPM', f"{int(last['RPM'])}")
        m_thr.metric('Throttle', f"{last['Throttle']:.1f} %")
        m_tmp.metric('Temp', f"{last['Temp']:.1f} °C")

        behavior_placeholder.subheader(last['Behavior_Label'])

        h_val = int(last['Health'])
        health_bar_placeholder.progress(h_val / 100, text=f"Oil Health: {h_val}%")

        temp = last['Temp']
        if temp < 70: thermal_status = '❄️ Cold Stress (Low Efficiency)'
        elif 85 <= temp <= 105: thermal_status = '✅ Optimal Operating Temperature'
        else: thermal_status = '🔥 Heat Stress (Rapid Degradation)'
        thermal_placeholder.warning(thermal_status)

        avg_stress = pd.to_numeric(st.session_state.history['Stress_Factor']).mean()
        current_health = last['Health']
        est_days = max(0, int((current_health / (0.5 * avg_stress)))) if avg_stress > 0 else 0
        f_date = (datetime.now() + timedelta(days=est_days)).strftime('%Y-%m-%d')
        forecast_placeholder.metric("Service Forecast", f_date, f"{est_days} Days")

        # 5-min averages logic - Fix: specify numeric_only and handle types
        df = st.session_state.history.copy()
        df['dt'] = pd.to_datetime(df['Time'], unit='s')
        numeric_cols = ['MAP', 'RPM', 'Throttle', 'Temp', 'Health']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        averages = df.set_index('dt')[numeric_cols].resample('5min').mean().tail(3)
        if not averages.empty:
            avg_placeholder.table(averages)

        chart_placeholder.line_chart(st.session_state.history.set_index('Time')['Health'])

if st.button('Start Sync Monitoring'):
    connection = obd.OBD('COM5', fast=True)
    if connection.is_connected():
        st.success('Behavior-Sync Online.')
        while True:
            r_map, r_rpm = connection.query(obd.commands.INTAKE_PRESSURE), connection.query(obd.commands.RPM)
            r_thr, r_tmp = connection.query(obd.commands.THROTTLE_POS), connection.query(obd.commands.COOLANT_TEMP)

            if all([not r.is_null() for r in [r_map, r_rpm, r_thr, r_tmp]]):
                v_map, v_rpm = float(r_map.value.magnitude), int(r_rpm.value.magnitude)
                v_thr, v_tmp = float(r_thr.value.magnitude), float(r_tmp.value.magnitude)
                mult, label = process_behavior(v_rpm)
                if v_tmp < 70 or v_tmp > 105: mult += 0.5
                features = np.array([[v_map, v_rpm, v_thr, v_tmp]])
                prediction = rf_model.predict(feature_scaler.transform(features))[0]
                health_score = max(0, min(100, 100 - (prediction * mult)))
                new_entry = pd.DataFrame([[time.time(), v_map, v_rpm, v_thr, v_tmp, health_score, mult, label]],
                                         columns=['Time', 'MAP', 'RPM', 'Throttle', 'Temp', 'Health', 'Stress_Factor', 'Behavior_Label'])
                st.session_state.history = pd.concat([st.session_state.history, new_entry], ignore_index=True)
                update_ui()
            time.sleep(0.5)
    else: st.error('OBD2 Connection Failed.')
