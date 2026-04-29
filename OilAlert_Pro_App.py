
import streamlit as st
import numpy as np
import joblib
import pandas as pd

st.set_page_config(page_title='OilAlert Pro', layout='wide')

# Professional UI Styling
st.markdown('''<style>
.main { background-color: #0e1117; }
.stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
</style>''', unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    # Loading the compressed .gz files specifically for GitHub/Streamlit Cloud compatibility
    model = joblib.load('oil_degradation_rf_model.pkl.gz')
    scaler = joblib.load('feature_scaler.pkl.gz')
    return model, scaler

try:
    rf_model, feature_scaler = load_assets()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

st.title('☢ OilAlert Pro: AI Engine Monitor')
st.divider()

params = st.query_params
if 'map' in params:
    try:
        v_map = float(params.get('map', 0))
        v_rpm = float(params.get('rpm', 0))
        v_thr = float(params.get('thr', 0))
        v_tmp = float(params.get('tmp', 0))

        features = np.array([[v_map, v_rpm, v_thr, v_tmp]])
        scaled_features = feature_scaler.transform(features)
        prediction = rf_model.predict(scaled_features)[0]
        health_score = max(0, min(100, 100 - prediction))

        cols = st.columns(4)
        cols[0].metric("MAP", f"{v_map} kPa")
        cols[1].metric("RPM", f"{int(v_rpm)}")
        cols[2].metric("Throttle", f"{v_thr}%")
        cols[3].metric("Coolant", f"{v_tmp} °C")

        st.markdown("### Oil Health Status")
        st.progress(health_score / 100)
        st.subheader(f"{health_score:.1f}% Health remaining")

        if v_tmp > 105 or v_tmp < 70:
            st.warning("⚠️ Thermal Stress Detected: Non-optimal operating temperature.")

    except Exception as e:
        st.error(f"Processing error: {e}")
else:
    st.info("🛰️ Listening for live data... Please run relay.py in your vehicle.")
