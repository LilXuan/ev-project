# # test_imports.py
# import sys
# sys.path.insert(0, '/home/axuan/bigdata/ev-project')

# print("Testing imports...")

# try:
#     from src.utils.spark import create_spark
#     print("✓ src.utils.spark")
# except Exception as e:
#     print(f"✗ src.utils.spark: {e}")

# try:
#     from src.utils.logger import get_logger
#     print("✓ src.utils.logger")
# except Exception as e:
#     print(f"✗ src.utils.logger: {e}")

# try:
#     from src.transforms.acn import run_pipeline
#     print("✓ src.transforms.acn")
# except Exception as e:
#     print(f"✗ src.transforms.acn: {e}")

# try:
#     from src.transforms.acn_validation import run_all_validations
#     print("✓ src.transforms.acn_validation")
# except Exception as e:
#     print(f"✗ src.transforms.acn_validation: {e}")

# print("\nAll imports checked!")


#!/usr/bin/env python3
from pyspark.sql import SparkSession

print("Starting Spark test...")
spark = SparkSession.builder \
    .appName("Test") \
    .master("local[1]") \
    .getOrCreate()

print(f"Spark version: {spark.version}")
df = spark.range(10)
print(f"Test DataFrame count: {df.count()}")
spark.stop()
print("Spark test completed successfully!")