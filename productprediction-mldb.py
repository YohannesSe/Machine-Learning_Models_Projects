# Databricks notebook source
# MAGIC %md
# MAGIC Import Libraries
# MAGIC

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, year, month, dayofmonth
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator

# COMMAND ----------

# MAGIC %md
# MAGIC Load data

# COMMAND ----------

df = spark.read.option("header", True).option("inferSchema", True).csv("dbfs:/FileStore/tables/ecommerce_sales_400kb.csv")
df.show(5)

# COMMAND ----------

df = df.withColumn("Year", year(col("Date"))) \
       .withColumn("Month", month(col("Date"))) \
       .withColumn("Day", dayofmonth(col("Date")))

# COMMAND ----------

df = df.drop("OrderID")

# COMMAND ----------

product_indexer = StringIndexer(inputCol="Product", outputCol="ProductIndex")
category_indexer = StringIndexer(inputCol="Category", outputCol="CategoryIndex")

# COMMAND ----------

assembler = VectorAssembler(
    inputCols=["ProductIndex", "CategoryIndex", "Year", "Month", "Day"],
    outputCol="features"
)

# COMMAND ----------

df.select("Date").distinct().show(100, truncate=False)
df.filter(col("Date").isNull()).count()

# COMMAND ----------

df = df.na.drop()

# COMMAND ----------

df = df.filter(col("Product").isNotNull() & col("Category").isNotNull())

# COMMAND ----------

from pyspark.ml.feature import StringIndexer

product_indexer = StringIndexer(inputCol="Product", outputCol="ProductIndex", handleInvalid="keep")
category_indexer = StringIndexer(inputCol="Category", outputCol="CategoryIndex", handleInvalid="keep")

# COMMAND ----------

# Clean nulls
df = df.filter(col("Product").isNotNull() & col("Category").isNotNull())

# Index with handleInvalid
product_indexer = StringIndexer(inputCol="Product", outputCol="ProductIndex", handleInvalid="keep")
category_indexer = StringIndexer(inputCol="Category", outputCol="CategoryIndex", handleInvalid="keep")

# COMMAND ----------

train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# COMMAND ----------

lr = LinearRegression(featuresCol="features", labelCol="Amount")

pipeline = Pipeline(stages=[product_indexer, category_indexer, assembler, lr])

# COMMAND ----------

model = pipeline.fit(train_df)

# COMMAND ----------

lr = LinearRegression(featuresCol="features", labelCol="Amount")

pipeline = Pipeline(stages=[product_indexer, category_indexer, assembler, lr])

# COMMAND ----------

model = pipeline.fit(train_df)

# COMMAND ----------

predictions = model.transform(test_df)

evaluator = RegressionEvaluator(labelCol="Amount", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print(f"RMSE: {rmse}")

r2 = evaluator.setMetricName("r2").evaluate(predictions)
print(f"R2: {r2}")

# COMMAND ----------

new_data = spark.createDataFrame([
    ("Product A", "Category 1", "2025-04-01")
], ["Product", "Category", "Date"])

new_data = new_data.withColumn("Date", col("Date").cast("date")) \
                   .withColumn("Year", year(col("Date"))) \
                   .withColumn("Month", month(col("Date"))) \
                   .withColumn("Day", dayofmonth(col("Date"))) \
                   .drop("Date")

predicted = model.transform(new_data)
predicted.select("Product", "Category", "prediction").show()

# COMMAND ----------

predictions = model.transform(test_df)
predictions.select("Date", "Amount", "prediction").show()

# COMMAND ----------

display(predictions.select("Date", "Amount", "prediction"))

# COMMAND ----------

import matplotlib.pyplot as plt

# Sort by date
pdf = pdf.sort_values(by="Date")

plt.figure(figsize=(12, 6))
plt.plot(pdf["Date"], pdf["Amount"], label="Actual Sales")
plt.plot(pdf["Date"], pdf["prediction"], label="Predicted Sales")
plt.xlabel("Date")
plt.ylabel("Sales Amount")
plt.title("Actual vs Predicted Sales")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# COMMAND ----------

# Convert predictions to Pandas DataFrame
pdf = predictions.select("Date", "Amount", "prediction").toPandas()

# COMMAND ----------

import matplotlib.pyplot as plt

pdf = pdf.sort_values(by="Date")  # Optional: ensure it's sorted

plt.figure(figsize=(12, 6))
plt.plot(pdf["Date"], pdf["Amount"], label="Actual Sales")
plt.plot(pdf["Date"], pdf["prediction"], label="Predicted Sales")
plt.xlabel("Date")
plt.ylabel("Sales Amount")
plt.title("Actual vs Predicted Sales")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %pip install seaborn

# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt

# Create scatter plot with regression line
plt.figure(figsize=(10, 6))
sns.regplot(data=pdf, x="Amount", y="prediction", line_kws={"color": "red"})

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Regression Line: Predicted vs Actual Sales")
plt.tight_layout()
plt.show()