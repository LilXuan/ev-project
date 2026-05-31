# configs/exp1_19mb.py
import os

APP_NAME = "ACN_Exp1_19MB"
DATASET_SIZE_MB = 19
INPUT_PATH = "hdfs://localhost:9000/ev-project/data/bronze/ev_sessions/caltech/*/*/*"
OUTPUT_PATH = "hdfs://localhost:9000/data/benchmark/acn/19mb"
RESULT_DIR = "results/loading"

# Thông số Spark tối ưu cho bộ data này
SPARK_CONF = {
    "spark.executor.memory": "2g",
    "spark.executor.cores": "1",
    "spark.sql.shuffle.partitions": "10" # Data nhỏ không cần chia quá nhiều partition
}
