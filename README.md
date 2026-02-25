### Dev/Creator: tubakhxn

# Spectral Market Decomposition Engine
<img width="2883" height="1491" alt="image" src="https://github.com/user-attachments/assets/f08fe44c-87ae-45fd-a9a4-c040970b4490" />

This project is a production-level Streamlit application for advanced frequency decomposition and spectral analysis of financial time series using Fast Fourier Transform (FFT).

## What is this project?

The **Spectral Market Decomposition Engine** allows users to:
- Analyze financial time series (simulated or real CSV data)
- Perform rolling window FFT to extract evolving frequency/amplitude structure
- Visualize the spectral surface in 3D and 2D (Plotly)
- Explore dominant frequencies and power spectrum interactively
- Use advanced controls for drift, volatility, window size, detrending, and more

It is designed for quantitative finance, signal processing, and scientific research.

## How to Fork and Run

1. **Fork this repository** on GitHub (click the "Fork" button at the top right of the repo page).
2. **Clone your fork** to your local machine:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Spectral-Market-Decomposition-Engine.git
   cd Spectral-Market-Decomposition-Engine
   ```
3. **(Optional) Create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   # Or manually:
   pip install streamlit numpy pandas plotly scipy
   ```
5. **Run the app:**
   ```bash
   streamlit run spectral_market_decomposition_engine.py
   ```
6. **Open your browser** to the local Streamlit URL (usually http://localhost:8501).

---

Enjoy exploring the spectral structure of financial markets!

