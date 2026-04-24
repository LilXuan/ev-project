# src/utils/spark.py
from pyspark.sql import SparkSession
from typing import Dict, Any


def create_spark(app_name: str, spark_conf: Dict[str, Any] = None) -> SparkSession:
    """
    Create and configure Spark session
    
    Args:
        app_name: Name of the Spark application
        spark_conf: Dictionary of Spark configuration parameters
    
    Returns:
        Configured SparkSession
    """
    builder = SparkSession.builder.appName(app_name)
    
    # Set default configurations for local development
    builder = builder.config("spark.sql.adaptive.enabled", "true")
    builder = builder.config("spark.sql.adaptive.coalescePartitions.enabled", "true")
    builder = builder.config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    builder = builder.config("spark.sql.parquet.compression.codec", "snappy")
    
    # Apply custom configurations
    if spark_conf:
        for key, value in spark_conf.items():
            builder = builder.config(key, value)
    
    # Create session
    spark = builder.getOrCreate()
    
    # Set log level
    spark.sparkContext.setLogLevel("WARN")
    
    return spark


def get_spark_session(app_name: str = "EV_Processing") -> SparkSession:
    """
    Get or create Spark session with default settings
    """
    return SparkSession.builder \
        .appName(app_name) \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()