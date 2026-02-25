import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from scipy.signal import detrend

# =============================
# Core Engine Functions
# =============================
def generate_gbm(n_steps, drift, volatility, dt, s0=100):
    np.random.seed(42)
    increments = np.random.normal(drift * dt, volatility * np.sqrt(dt), n_steps)
    log_prices = np.cumsum(increments) + np.log(s0)
    prices = np.exp(log_prices)
    return prices

def compute_returns(prices, log_return=True):
    if log_return:
        return np.diff(np.log(prices))
    else:
        return np.diff(prices)

def rolling_fft(series, window_size, sampling_freq, detrend_data=False):
    n = len(series)
    n_windows = n - window_size + 1
    freq_matrix = []
    amp_matrix = []
    power_matrix = []
    for i in range(n_windows):
        window = series[i:i+window_size]
        if detrend_data:
            window = detrend(window)
        fft_vals = np.fft.rfft(window)
        fft_freqs = np.fft.rfftfreq(window_size, d=1/sampling_freq)
        amplitude = np.abs(fft_vals)
        power = amplitude ** 2
        freq_matrix.append(fft_freqs)
        amp_matrix.append(amplitude)
        power_matrix.append(power)
    freq_matrix = np.array(freq_matrix)
    amp_matrix = np.array(amp_matrix)
    power_matrix = np.array(power_matrix)
    return freq_matrix, amp_matrix, power_matrix

def build_spectral_surface(amp_matrix):
    return amp_matrix.T

# =============================
# Streamlit Dashboard
# =============================
st.set_page_config(page_title="Spectral Market Decomposition Engine", layout="wide", page_icon="📈")
st.title("Spectral Market Decomposition Engine")

with st.sidebar:
    st.header("Data Source")
    data_mode = st.selectbox("Choose Data Mode", ["Simulated", "Upload CSV"])
    drift = st.slider("Drift (μ)", -0.05, 0.05, 0.01, 0.001)
    volatility = st.slider("Volatility (σ)", 0.01, 0.5, 0.1, 0.01)
    window_size = st.slider("Rolling Window Size", 64, 1024, 256, 8)
    sampling_freq = st.number_input("Sampling Frequency (Hz)", min_value=1.0, value=1.0, step=0.1)
    detrend_data = st.toggle("Detrend Data", value=False)
    log_return = st.toggle("Log Returns", value=True)
    heatmap_mode = st.toggle("Heatmap Mode", value=False)

if data_mode == "Simulated":
    n_steps = 4096
    prices = generate_gbm(n_steps, drift, volatility, 1/sampling_freq)
    time = np.arange(n_steps)
    df = pd.DataFrame({"Time": time, "Price": prices})
else:
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if 'time' in df.columns and 'price' in df.columns:
            time = df['time'].values
            prices = df['price'].values
        else:
            st.error("CSV must have 'time' and 'price' columns.")
            st.stop()
    else:
        st.warning("Upload a CSV file to proceed.")
        st.stop()

returns = compute_returns(prices, log_return=log_return)

freq_matrix, amp_matrix, power_matrix = rolling_fft(
    returns, window_size, sampling_freq, detrend_data=detrend_data
)
spectral_surface = build_spectral_surface(amp_matrix)
time_surface = time[window_size:]
frequency_surface = freq_matrix[0]

# =============================
# Visualization
# =============================
# 3D Spectral Surface
surface_fig = go.Figure(
    data=[
        go.Surface(
            x=time_surface,
            y=frequency_surface,
            z=spectral_surface,
            colorscale="Viridis",
            showscale=True,
            opacity=0.95,
        )
    ],
    layout=go.Layout(
        title="Rolling Spectral Surface",
        scene=dict(
            xaxis_title="Time",
            yaxis_title="Frequency",
            zaxis_title="Amplitude",
            bgcolor="#181818",
            xaxis=dict(showgrid=True, gridcolor="#333", zeroline=False),
            yaxis=dict(showgrid=True, gridcolor="#333", zeroline=False),
            zaxis=dict(showgrid=True, gridcolor="#333", zeroline=False),
        ),
        template="plotly_dark",
        margin=dict(l=0, r=0, b=0, t=40),
    )
)
surface_fig.update_traces(
    contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True)
)
surface_fig.update_layout(
    scene_camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
    updatemenus=[
        dict(
            type="buttons",
            showactive=False,
            buttons=[
                dict(
                    label="Rotate",
                    method="animate",
                    args=[None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True}]
                )
            ],
            x=0.1,
            y=0.1
        )
    ]
)

# 2D FFT Spectrum Plot (last window)
last_amp = amp_matrix[-1]
last_freq = freq_matrix[-1]
dominant_idx = np.argmax(last_amp)
dominant_freq = last_freq[dominant_idx]
dominant_amp = last_amp[dominant_idx]
spectrum_fig = go.Figure(
    data=[
        go.Scatter(
            x=last_freq,
            y=last_amp,
            mode="lines+markers",
            marker=dict(color="#00FFAA"),
            line=dict(color="#00FFAA"),
            name="Amplitude"
        ),
        go.Scatter(
            x=[dominant_freq],
            y=[dominant_amp],
            mode="markers",
            marker=dict(size=12, color="red"),
            name="Dominant Frequency"
        )
    ],
    layout=go.Layout(
        title="FFT Spectrum (Last Window)",
        xaxis_title="Frequency",
        yaxis_title="Amplitude",
        template="plotly_dark",
        margin=dict(l=40, r=40, b=40, t=40),
    )
)

# Power Spectrum (last window)
power_fig = go.Figure(
    data=[
        go.Bar(
            x=last_freq,
            y=power_matrix[-1],
            marker=dict(color="#1f77b4"),
            name="Power"
        )
    ],
    layout=go.Layout(
        title="Power Spectrum (Last Window)",
        xaxis_title="Frequency",
        yaxis_title="Power",
        template="plotly_dark",
        margin=dict(l=40, r=40, b=40, t=40),
    )
)

# Heatmap mode
if heatmap_mode:
    heatmap_fig = go.Figure(
        data=[
            go.Heatmap(
                x=time_surface,
                y=frequency_surface,
                z=spectral_surface,
                colorscale="Viridis",
                showscale=True,
            )
        ],
        layout=go.Layout(
            title="Spectral Intensity Heatmap",
            xaxis_title="Time",
            yaxis_title="Frequency",
            template="plotly_dark",
            margin=dict(l=40, r=40, b=40, t=40),
        )
    )

# =============================
# Main Layout
# =============================
st.markdown("---")
col1, col2 = st.columns([2, 1])

with col1:
    st.plotly_chart(surface_fig, use_container_width=True)
    if heatmap_mode:
        st.plotly_chart(heatmap_fig, use_container_width=True)

with col2:
    st.metric("Dominant Frequency (Hz)", f"{dominant_freq:.4f}")
    st.metric("Dominant Amplitude", f"{dominant_amp:.4f}")
    st.plotly_chart(spectrum_fig, use_container_width=True)
    st.plotly_chart(power_fig, use_container_width=True)

st.markdown("<style>body { background-color: #181818; color: #EEE; } .stMetric { font-size: 1.2em; } .stPlotlyChart { border-radius: 8px; }</style>", unsafe_allow_html=True)

# =============================
# Caching for Performance
# =============================
@st.cache_data(show_spinner=False)
def cached_gbm(n_steps, drift, volatility, dt, s0=100):
    return generate_gbm(n_steps, drift, volatility, dt, s0)

@st.cache_data(show_spinner=False)
def cached_fft(series, window_size, sampling_freq, detrend_data):
    return rolling_fft(series, window_size, sampling_freq, detrend_data)

# =============================
# End of File
# =============================
