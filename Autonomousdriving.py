# Databricks notebook source
# MAGIC %md
# MAGIC ## Autonomous Driving Car Steering Angle & Acceleration Prediction Model
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Upload Libraries and Dataset 

# COMMAND ----------

from pyspark.sql import SparkSession

from pyspark.sql.functions import col, sum

from pyspark.ml.feature import (
    StringIndexer, 
    OneHotEncoder, 
    VectorAssembler, 
    MinMaxScaler, 
    StandardScaler,
    )

from pyspark.ml import Pipeline
from sklearn.model_selection import train_test_split
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# COMMAND ----------

df = spark.read.csv("/FileStore/tables/autonomous_driving.csv", header=True, inferSchema=True)
df.show()

# COMMAND ----------

#count missing values for each column by creating a df with null counts for each column
missing_values = df.select([sum(col(c).isNull().cast("int")).alias(c) for c in df.columns])
missing_values.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Wrangling

# COMMAND ----------

#Check the schema
df.printSchema
df.printSchema()

#Step 1-index the weather and lane column(convert it to numerical indices)
weather_indexer = StringIndexer(inputCol="weather_condition", outputCol="weather_conditionIndex")
lane_indexer = StringIndexer(inputCol="lane_position", outputCol="lane_positionIndex")
#Step 2-apply OneHotEncoder to the indexed column
encoder = OneHotEncoder(inputCols=["weather_conditionIndex", "lane_positionIndex"], outputCols=["weather_conditionVec", "lane_positionVec"])


# COMMAND ----------

from pyspark.sql.functions import hour, to_timestamp

df = df.withColumn("hour", hour(to_timestamp("timestamp")))


# COMMAND ----------

# Step 1: Define numerical features
numerical_features = ["hour", "speed_kmh", "location_lat", "location_lon", "obstacles_detected"]
# Step 2: Combine all features into one-
num_assembler= VectorAssembler(inputCols=numerical_features, outputCol="numerical_features")

# COMMAND ----------

#comibine one-hot encoded features and scaled numerical features into one feature column
final_assembler = VectorAssembler(inputCols=["weather_conditionVec", "lane_positionVec", "numerical_features"], outputCol="features")

# COMMAND ----------

# step: build and run a pipeline model will all stages
pipeline = Pipeline(stages = [weather_indexer, lane_indexer, encoder, num_assembler, final_assembler])
#step: fit the pipeline on the df
pipeline_model=pipeline.fit(df)
final_df=pipeline_model.transform(df)

#select features and target (label) column
final_df = final_df.select("features", "steering_angle_deg", "acceleration_mps2")
final_df.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Split Data and Train the Model

# COMMAND ----------

from pyspark.ml.regression import LinearRegression

# Model 1: Predict steering_angle_deg
steering_model = LinearRegression(featuresCol="features", labelCol="steering_angle_deg")
steering_trained = steering_model.fit(final_df)
steering_pred = steering_trained.transform(final_df).withColumnRenamed("prediction", "pred_steering_angle")

# Model 2: Predict acceleration_mps2
accel_model = LinearRegression(featuresCol="features", labelCol="acceleration_mps2")
accel_trained = accel_model.fit(final_df)
accel_pred = accel_trained.transform(final_df).withColumnRenamed("prediction", "pred_acceleration")


# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id

steering_pred = steering_pred.withColumn("id", monotonically_increasing_id())
accel_pred = accel_pred.withColumn("id", monotonically_increasing_id())

combined_df = steering_pred.join(accel_pred.select("id", "pred_acceleration"), on="id").drop("id")
combined_df.select("features", "pred_steering_angle", "pred_acceleration").show(5)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Model Evalaution

# COMMAND ----------

evaluator_steering = RegressionEvaluator(
    labelCol="steering_angle_deg", predictionCol="pred_steering_angle", metricName="rmse")
rmse_steering = evaluator_steering.evaluate(steering_pred)

evaluator_steering_mae = RegressionEvaluator(
    labelCol="steering_angle_deg", predictionCol="pred_steering_angle", metricName="mae")
mae_steering = evaluator_steering_mae.evaluate(steering_pred)

evaluator_steering_r2 = RegressionEvaluator(
    labelCol="steering_angle_deg", predictionCol="pred_steering_angle", metricName="r2")
r2_steering = evaluator_steering_r2.evaluate(steering_pred)


# COMMAND ----------

print(f"🚗 Steering Angle Evaluation:")
print(f"  RMSE: {rmse_steering:.3f}")
print(f"  MAE:  {mae_steering:.3f}")
print(f"  R²:   {r2_steering:.3f}")

print(f"\n🏎️ Acceleration Evaluation:")
print(f"  RMSE: {rmse_accel:.3f}")
print(f"  MAE:  {mae_accel:.3f}")
print(f"  R²:   {r2_accel:.3f}")


# COMMAND ----------

# MAGIC %md
# MAGIC ### Scatter Plotting

# COMMAND ----------

# Convert to Pandas (small dataset recommended)
steering_pd = steering_pred.select("steering_angle_deg", "pred_steering_angle").toPandas()
accel_pd = accel_pred.select("acceleration_mps2", "pred_acceleration").toPandas()


# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(14, 6))

# Plot 1: Steering Angle
plt.subplot(1, 2, 1)
sns.scatterplot(data=steering_pd, x="steering_angle_deg", y="pred_steering_angle", alpha=0.6)
plt.plot([steering_pd.steering_angle_deg.min(), steering_pd.steering_angle_deg.max()],
         [steering_pd.steering_angle_deg.min(), steering_pd.steering_angle_deg.max()],
         color='red', linestyle='--')  # Line y=x
plt.title("Steering Angle: Actual vs Predicted")
plt.xlabel("Actual")
plt.ylabel("Predicted")

# Plot 2: Acceleration
plt.subplot(1, 2, 2)
sns.scatterplot(data=accel_pd, x="acceleration_mps2", y="pred_acceleration", alpha=0.6)
plt.plot([accel_pd.acceleration_mps2.min(), accel_pd.acceleration_mps2.max()],
         [accel_pd.acceleration_mps2.min(), accel_pd.acceleration_mps2.max()],
         color='red', linestyle='--')  # Line y=x
plt.title("Acceleration: Actual vs Predicted")
plt.xlabel("Actual")
plt.ylabel("Predicted")

plt.tight_layout()
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ### Analaysis of Residual Error 

# COMMAND ----------

# For steering angle
steering_pd["residual"] = steering_pd["steering_angle_deg"] - steering_pd["pred_steering_angle"]

# For acceleration
accel_pd["residual"] = accel_pd["acceleration_mps2"] - accel_pd["pred_acceleration"]


# COMMAND ----------

plt.figure(figsize=(14, 6))

# Residuals for Steering Angle
plt.subplot(1, 2, 1)
sns.scatterplot(x=steering_pd["pred_steering_angle"], y=steering_pd["residual"], alpha=0.6)
plt.axhline(0, color="red", linestyle="--")
plt.title("Residuals: Steering Angle")
plt.xlabel("Predicted Steering Angle")
plt.ylabel("Residual (Actual - Predicted)")

# Residuals for Acceleration
plt.subplot(1, 2, 2)
sns.scatterplot(x=accel_pd["pred_acceleration"], y=accel_pd["residual"], alpha=0.6)
plt.axhline(0, color="red", linestyle="--")
plt.title("Residuals: Acceleration")
plt.xlabel("Predicted Acceleration")
plt.ylabel("Residual (Actual - Predicted)")

plt.tight_layout()
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Contribution in Model Prediction

# COMMAND ----------

from pyspark.ml.regression import LinearRegression

# Initialize and fit the model
lr = LinearRegression(featuresCol="features", labelCol="steering_angle_deg")
lr_model = lr.fit(final_df)

# Access the coefficients (weights) and intercept
coeffs = lr_model.coefficients
intercept = lr_model.intercept

# Print the intercept and coefficients
print(f"Intercept: {intercept:.4f}")
print("Feature Coefficients:")
for name, coef in zip(final_assembler.getInputCols(), coeffs):
    print(f"{name}: {coef:.4f}")


# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

# Combine feature names and their corresponding coefficients
feature_names = final_assembler.getInputCols()
coef_dict = dict(zip(feature_names, coeffs))

# Sort features by their absolute contribution
sorted_features = sorted(coef_dict.items(), key=lambda x: abs(x[1]), reverse=True)

# Plotting
plt.figure(figsize=(10, 6))
sns.barplot(x=[item[0] for item in sorted_features], y=[item[1] for item in sorted_features])
plt.xticks(rotation=90)
plt.title("Feature Contributions (Coefficients) in Linear Regression")
plt.xlabel("Feature")
plt.ylabel("Coefficient Value")
plt.show()


# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

# Assuming you have already trained the linear regression model (lr_model) and have the coefficients

# Get all feature names (combining both numeric and categorical feature names)
numeric_feature_names = ["hour", "speed_kmh", "location_lat", "location_lon", "obstacles_detected"]
encoded_feature_names = encoder.getOutputCols()  # Get names of encoded categorical features

# Combine numeric and encoded feature names
all_feature_names = numeric_feature_names + encoded_feature_names

# Get the coefficients for all features (numeric + categorical)
coeffs = lr_model.coefficients

# Combine feature names and their corresponding coefficients into a dictionary
coef_dict = dict(zip(all_feature_names, coeffs))

# Sort features by their absolute contribution (for better visual interpretation)
sorted_features = sorted(coef_dict.items(), key=lambda x: abs(x[1]), reverse=True)

# Plotting the feature contributions (coefficients) for all features
plt.figure(figsize=(12, 6))
sns.barplot(x=[item[0] for item in sorted_features], y=[item[1] for item in sorted_features])
plt.xticks(rotation=90)  # Rotate the feature names for better readability
plt.title("Feature Contributions (Coefficients) in Linear Regression for All Features")
plt.xlabel("Feature")
plt.ylabel("Coefficient Value")
plt.show()


# COMMAND ----------

import logging
from datetime import datetime

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Generate a unique timestamp for the model's save path
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Correct path to save the model without /dbfs/
model_folder = f"dbfs:/FileStore/tables/autonomous/model_{timestamp}"

# Save the model
lr_model.write().save(model_folder)

# Log the model save action
logger.info(f"Linear Regression model saved successfully.\n"
             f"Timestamp: {timestamp}\n"
             f"Model folder path: {model_folder}\n"
             f"Use this path to load or manage your model.")


# COMMAND ----------


loaded_model = LinearRegressionModel.load(model_folder)


# COMMAND ----------

# Check if folder exists
dbutils.fs.ls("dbfs:/FileStore/tables/autonomous/")


# COMMAND ----------

#copy the model to local file system
dbutils.fs.cp("dbfs:/FileStore/tables/autonomous/model_20250418_205654", "file:/tmp/model_{timestamp}", True)

