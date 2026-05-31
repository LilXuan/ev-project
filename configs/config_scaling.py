# configs/config_scaling.py
import os
from pathlib import Path

APP_NAME = "ACN_Dataset_Scaling_1GB"

# Hệ số mặc định nếu không truyền qua CLI
DEFAULT_SCALE_FACTOR = 50 
DEFAULT_PARTITIONS = 12

# Đường dẫn dữ liệu trên HDFS
INPUT_PATH = "hdfs://localhost:9000/ev-project/data/bronze/ev_sessions/caltech/*/*/*"
OUTPUT_BASE_PATH = "hdfs://localhost:9000/data/benchmark/acn/1gb"

# Quản lý đường dẫn kết quả nội bộ dự án
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = os.path.join(str(PROJECT_ROOT), "results", "scaling")
os.makedirs(RESULT_DIR, exist_ok=True)

# === BỔ SUNG THÊM CÁC DÒNG NÀY VÀO FILE CONFIG CỦA BẠN ===
SCALING_METRICS_CSV = os.path.join(RESULT_DIR, "1gb_scaling_metrics.csv")
SCALING_VALIDATION_CSV = os.path.join(RESULT_DIR, "1gb_scaling_validation.csv")
# ========================================================

# Cấu hình tài nguyên Spark Cluster
SPARK_CONF = {
    "spark.executor.memory": "1g",
    "spark.driver.memory": "1g",
    "spark.sql.shuffle.partitions": "12",
    "spark.default.parallelism": "12"
}