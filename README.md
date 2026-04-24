# Solar-Power-Forecasting-for-Battery-Energy-Storage-Scheduling

**CE 295: Data Science for Energy | UC Berkeley**

## 🎯 Project Overview
This project aims to optimize Battery Energy Storage System (BESS) operations by integrating high-precision solar power forecasting. We bridge the gap between **Deep Learning** (for generation prediction) and **Linear Programming** (for optimal dispatch) to reduce grid dependency and energy costs.

### Key Objectives:
- **Forecast:** Develop an LSTM-based neural network to predict hourly solar PV generation.
- **Optimize:** Formulate a deterministic Linear Program (LP) to schedule battery charge/discharge cycles.
- **Validate:** Compare forecast-driven strategies against naive baselines through historical backtesting.

## 🛠️ Tech Stack
- **Languages:** Python (Advanced)
- **Machine Learning:** PyTorch (LSTM), Scikit-Learn (ARX Baseline, Ridge)
- **Data Science:** Pandas, NumPy, Matplotlib (EDA & Signal Processing)
- **Optimization:** Linear Programming (Optimization of state-of-charge constraints)
- **Source Data:** NREL National Solar Radiation Database (NSRDB) - Los Angeles Area.

## 📊 Methodology

### 1. Data Pipeline & EDA
- Ingestion of multi-year satellite weather data (GHI, DNI, DHI, Temperature, Humidity).
- Feature engineering: Time-cyclic encoding (Sine/Cosine for hours/months), clear-sky normalization.
- Data cleaning: Savitzky-Golay filtering for signal smoothing and handling missing telemetry.

### 2. Forecasting Models
- **Baseline (ARX):** AutoRegressive model with eXogenous inputs for benchmarking.
- **Deep Learning (LSTM):** A 2-layer Long Short-Term Memory network designed to capture temporal dependencies in weather patterns.
  - *Metric:* Targeted reduction in RMSE and MAE compared to persistence models.

### 3. BESS Optimization
- Objective function: Minimize grid energy costs over a 24h horizon.
- Constraints: Battery capacity, power rating (C-rate), and state-of-charge (SoC) limits.

## 🚀 Installation & Usage
1. Clone the repo: `git clone https://github.com/matthieugomez31/solar-bess-forecasting.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the data pipeline: `python src/data_pipeline.py`
4. Train the LSTM: `python src/models/lstm_model.py`

## 👥 Team Members
- Matthieu Gomez
- François Cacheux
- Sohel Dinnoo
- Marcu-Andria Castelli
- Antoine Tortochaux
- Solon Palu.

---
*Developed as part of the CE 295 course.*