# 🚴‍♂️ Bike Rental Demand Prediction using Azure ML Designer

This project predicts the **daily count of bike rentals** using regression models built entirely with **Azure Machine Learning Designer** — a no-code visual ML environment. It demonstrates a real-world use case of demand forecasting based on weather, seasonality, and calendar data.

---

## 📌 Problem Statement

Predict the total number of daily bike rentals (`cnt`) using features such as temperature, humidity, season, and whether it's a working day or holiday.

---

## 📊 Dataset

- **Name**: Bike Sharing Dataset (Day)
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset)
- **Records**: 731 days (2 years)
- **Features**: Weather, date, season, holiday, working day, temp, humidity, windspeed

---

## 🧪 Pipeline Overview (Azure ML Designer)

![Pipeline Diagram] (pipeline-diagram.png)

### Modules Used:
- **Import Data**: Load the dataset (bike_day.csv)
- **Select Columns**: Drop `instant`, `casual`, `registered`, `dteday`
- **Edit Metadata**: Mark categorical features
- **Convert to Indicator Values**: One-hot encode categorical columns
- **Execute Python Script**: Add derived features (`avg_temp`, `is_weekend`)
- **Split Data**: 80% Train / 20% Test
- **Train Model**: Boosted Decision Tree Regression
- **Evaluate Model**: MAE, RMSE, R²

---

## 🧠 Model Info

- **Model Type**: Boosted Decision Tree Regression
- **Target Variable**: `cnt` (Total bike rentals)

---

## 📈 Results

| Metric | Value |
|--------|-------|
| MAE    | 562.85 |
| RMSE   | 757.10 |
| R²     | 0.86 |
| RSE    | 0.13 |
| RAE    | 0.32 |

---

## 🧰 Azure Services Used

- **Azure Machine Learning Designer**
- **Azure Machine Learning Studio Workspace**
- **Execute Python Script module**
- (Optional) Azure Blob Storage for data input/output

---

## 🧾 How to Run

1. Clone this repo
2. Upload `bike_day.csv` into your Azure ML Studio workspace
3. Rebuild the pipeline using the diagram and steps above
4. Run the pipeline and view evaluation metrics

---

## 📄 License

MIT License. Free to use and modify for educational purposes.

---
