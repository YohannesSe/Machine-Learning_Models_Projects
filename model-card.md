# 🧾 Model Card: Bike Rental Demand Predictor

## Model Overview

This regression model predicts the total number of bike rentals (`cnt`) for a given day based on weather, season, and calendar data. Built using **Boosted Decision Tree Regression** in Azure ML Designer.

---

## Intended Use

- Demand forecasting for bike-sharing systems
- Urban transportation planning
- Weather-aware scheduling of bikes

---

## Input Features

| Feature | Description |
|---------|-------------|
| season | 1: Spring, 2: Summer, 3: Fall, 4: Winter |
| yr | Year (0: 2011, 1: 2012) |
| mnth | Month (1 to 12) |
| holiday | Is it a holiday (0/1) |
| weekday | Day of week (0 = Sunday) |
| workingday | Is it a working day (0/1) |
| weathersit | Weather condition category (1–4) |
| temp | Normalized temperature |
| atemp | Normalized "feels like" temp |
| hum | Humidity |
| windspeed | Wind speed |
| avg_temp | Derived: average of temp & atemp |
| is_weekend | Derived: 1 if Saturday/Sunday |

---

## Target Variable

- `cnt` — Total bike rentals (casual + registered)

---

## Evaluation Metrics

| Metric | Value |
|--------|-------|
| MAE    | 562.85 |
| RMSE   | 757.10 |
| R²     | 0.86 |
| RSE    | 0.13 |
| RAE    | 0.32 |

---

## Ethical Considerations

- No personal or sensitive data included.
- Model may reflect bias due to seasonality or region-specific patterns.

---

## Limitations

- Based on historical data from Washington, D.C.
- Doesn’t consider real-time traffic or events
- Can underperform during extreme weather or holidays

---

## Authors

Created by Yohannes Setotaw using Azure ML Designer.

---

