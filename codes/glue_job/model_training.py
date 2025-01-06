import sys
from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
import boto3
import os
import logging

# Initialize Spark session
spark = SparkSession.builder \
    .appName("GlueModelTraining") \
    .getOrCreate()

# Load sample data (MNIST dataset in LibSVM format)
region = "us-east-1"
data_path = f"s3a://sagemaker-sample-data-{region}/spark/mnist/train/"
training_data = spark.read.format("libsvm") \
    .option("numFeatures", "784") \
    .load(data_path)

logging.info("Training data loaded")
training_data.show()

# Prepare data for K-Means
assembler = VectorAssembler(inputCols=["features"], outputCol="features_vector")
training_data = assembler.transform(training_data)

logging.info("Data prepared for K-Means")
training_data.show()

# Train a K-Means model locally
kmeans = KMeans(k=10, seed=42, featuresCol="features_vector", predictionCol="cluster")
model = kmeans.fit(training_data)

# Save the trained model locally
local_model_path = "/tmp/kmeans_model"
model.save(local_model_path)

# Upload the model artifacts to S3
s3_bucket = "textclassificationmldemo-model-archiving-us-east-1-1272"  # Replace with your S3 bucket name
s3_prefix = "models/kmeans_model"  # Replace with your desired S3 prefix

s3_client = boto3.client("s3")

# Upload all files in the local model directory to S3
for root, dirs, files in os.walk(local_model_path):
    for file in files:
        local_file_path = os.path.join(root, file)
        s3_key = os.path.join(s3_prefix, os.path.relpath(local_file_path, local_model_path))
        s3_client.upload_file(local_file_path, s3_bucket, s3_key)

print(f"Model artifacts uploaded to s3://{s3_bucket}/{s3_prefix}")

# Stop the Spark session
spark.stop()