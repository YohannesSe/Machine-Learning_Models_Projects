🚗 Autonomous Driving Simulation: Steering & Acceleration Prediction
📌 Project Overview
This project demonstrates a machine learning-based simulation of an autonomous vehicle that predicts steering angle and acceleration using environmental and sensor data. It utilizes regression models trained and evaluated in an Azure ML Notebook environment.
📊 Dataset Features
vehicle_id, timestamp, speed_kmh, acceleration_mps2, steering_angle_deg, location_lat, location_lon, obstacles_detected, weather_condition, lane_position

🎯 Target Variables
steering_angle_deg (continuous) and acceleration_mps2 (continuous)

⚙️ Model Workflow
Data Preprocessing
Convert timestamp to hour
Encode categorical features (weather_condition, lane_position)

Feature Engineering
Extract hour from timestamp
One-hot encode conditions
Model Training
Multi-output linear regression using MultiOutputRegressor
Evaluation
MAE, RMSE, R², RSE, RAE
Scatter + regression plots of actual vs predicted values
Model Export
Save trained model with joblib
📈 Model Performance
| Metric | Value |
|--------|-------|
| MAE    | 0.520 |
| RMSE   | 0.7851|
| R²     | 0.97 |
🧪 Inference Example
python
Copy
Edit
sample_input = X_test.iloc[[0]]
prediction = model.predict(sample_input)
print("Predicted Steering & Acceleration:", prediction)

🛠️ Tools & Libraries
Python, Pandas, Scikit-learn, Seaborn, Matplotlib
Azure ML Notebook Environment

✍️ Author
Yohannes
Project developed in Azure ML Designer & Notebooks
📧 [LinkedIn]
