
# Blood Inventory Demand Forecasting — LSTM Deep Learning

Predicting daily blood bag demand using LSTM deep learning to help blood banks maintain optimal inventory, reduce wastage, and prevent shortages.

Built using real blood bank transaction data (6,894 records across 8 blood groups, June 2020 – January 2021).

---

## The Problem

Blood banks face a difficult inventory challenge: donate too little and you risk shortages during emergencies; donate too much and bags expire unused. Daily demand is highly variable — in this dataset it ranged from 1 to 87 bags per day. Accurate demand forecasting allows blood banks to plan donation drives proactively rather than reactively.

---

## What This Project Does

- **Data cleaning and preparation** — parses transaction records, handles missing values, fills date gaps, and transforms bag-level records into daily demand time series
- **Exploratory data analysis** — demand distribution by blood group, day-of-week patterns, shortage risk identification
- **Feature engineering** — sliding window sequences, rolling averages, temporal features
- **LSTM model** — two-layer LSTM with dropout, trained with early stopping, evaluated against a naive baseline
- **Evaluation** — MAE and RMSE on holdout test set, compared against naive (yesterday's demand) baseline
- **Shortage risk analysis** — identifies high-demand days exceeding 1.5x average

---

## Key Findings

- **O Positive** accounts for ~36% of all issues — highest priority for inventory management at ~8 bags/day
- Daily demand is **highly variable** (std dev ~14 bags/day) — forecasting model meaningfully outperforms naive baseline
- ~15-20% of days exceed 1.5x average demand — identifying these in advance allows pre-emptive collection drives
- The hardest part was **data preparation**, not the model — inconsistent records, date gaps, and mixed null patterns in the Issued/Discarded columns required careful handling

---

## Stack

- Python (Pandas, NumPy, TensorFlow/Keras, Scikit-learn, Matplotlib, Seaborn)
- LSTM architecture: 64 units → Dropout(0.2) → 32 units → Dropout(0.2) → Dense(16) → Dense(1)
- Evaluation: MAE, RMSE on 30-day holdout test set
- Baseline comparison: naive forecast (previous day's demand)

---

## Dataset

Real blood bank transaction records — 6,894 bags across 8 blood groups.
Columns: `Bag Name`, `Blood Group`, `Donated`, `Issued`, `Discarded`

---

## Project Structure

```
blood-inventory-lstm/
├── blood_inventory_lstm.ipynb   # Main notebook — full pipeline
├── list.xls                     # Dataset (CSV format despite .xls extension)
├── plots/                       # Generated visualisations
│   ├── 01_blood_group_distribution.png
│   ├── 02_daily_demand_timeseries.png
│   ├── 03_demand_by_day_of_week.png
│   ├── 04_training_history.png
│   ├── 05_forecast_vs_actual.png
│   ├── 06_demand_by_blood_group.png
│   └── 07_shortage_risk.png
└── README.md
```

---

## How to Run

```bash
pip install pandas numpy tensorflow scikit-learn matplotlib seaborn
jupyter notebook blood_inventory_lstm.ipynb
```

Update the data path in Section 1 to point to your copy of `list.xls`.

---

## Skills Demonstrated

`LSTM` `Deep Learning` `TensorFlow/Keras` `Time-Series Forecasting` `Feature Engineering` `Data Cleaning` `ETL Pipeline` `Model Evaluation` `Python` `Healthcare Analytics`
