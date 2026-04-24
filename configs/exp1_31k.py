# configs/exp1_31k.py
import os

APP_NAME = "ACN_Exp1_31k"
INPUT_PATH = "hdfs://localhost:9000/ev-project/data/bronze/ev_sessions/caltech/*/*/*"
OUTPUT_PATH = "hdfs://localhost:9000/ev-project/data/silver/ev_sessions/caltech"
METRICS_LOG = "experiment_results.csv"

# Thông số Spark tối ưu cho bộ data này
SPARK_CONF = {
    "spark.executor.memory": "2g",
    "spark.executor.cores": "1",
    "spark.sql.shuffle.partitions": "10" # Data nhỏ không cần chia quá nhiều partition
}