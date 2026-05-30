# configs/config_caltech.py
import os
from pathlib import Path

APP_NAME = "ACN_Caltech_Preprocessing_Local"

#Cấu hình dynamic size label (phục vụ tự động nhận diện dung lượng đầu vào/đầu ra)
DATASET_SIZE_MB = 1024  # Hoặc đổi thành 500 nếu test với tập 500MB
SIZE_LABEL = "1gb" if DATASET_SIZE_MB == 1024 else "500mb"
# # Input & Output Paths trên HDFS định cấu hình đồng bộ
# INPUT_PATH = f"hdfs://localhost:9000/data/benchmark/acn/{SIZE_LABEL}/"
# OUTPUT_PATH = f"hdfs://localhost:9000/data/benchmark/acn/{SIZE_LABEL}_processed/"

INPUT_PATH = "hdfs://localhost:9000/data/benchmark/acn/500mb/"
OUTPUT_PATH = "hdfs://localhost:9000/ev-project/data/silver/ev_sessions/500mb"


# === ĐỒNG BỘ CẤU TRÚC ĐƯỜNG DẪN METRICS THEO CHUẨN MỚI ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = os.path.join(str(PROJECT_ROOT), "results", "processing")
os.makedirs(RESULT_DIR, exist_ok=True)

METRICS_LOG = os.path.join(RESULT_DIR, "processing_metrics.csv")
RESEARCH_LOG = os.path.join(RESULT_DIR, "raw_metrics", f"{APP_NAME}.json")
# ========================================================

TRAINING_METRICS_LOG = str(PROJECT_ROOT / "metrics" / "training_metrics.csv")

# Processing configuration
NUM_PARTITIONS = 8
FAIL_ON_VALIDATION_ERROR = False

# Feature configuration
FEATURES = [
    'hour', 'day_of_week', 'month', 'season',
    'duration', 'charging_duration', 'charging_duration_log',
    'hour_sin', 'hour_cos', 'day_of_year', 'week_of_year', 'is_holiday',
    'lag_1_log', 'lag_2_log', 'lag_3_log',
    'rolling_mean_3_log', 'rolling_mean_5_log'
]

CRITICAL_COLS = ['kWhDelivered', 'duration', 'charging_duration_log']
OUTLIER_COLS = ["duration", "charging_duration", "kWhDelivered"]
LAGS = (1, 2, 3)
WINDOWS = (3, 5)

# Training configuration
TRAIN_CONFIG = {
    "train_fraction": 0.8,
    "seed": 42,
    "max_depth": 10,
    "max_iter": 100,
    "feature_cols": FEATURES,
    "target_col": "kWhDelivered_log"
}

USE_PAPER_SUBSET = False
PAPER_SUBSET_SIZE = 14496
CHECKPOINT_PATH = "/tmp/acn_checkpoint"

SPARK_CONF = {
    "spark.executor.memory": "1g",
    "spark.executor.cores": "2",
    "spark.default.parallelism": "8",
    "spark.sql.shuffle.partitions": "8",
    "spark.driver.memory": "1g",
    "spark.metrics.namespace": "driver",
    "spark.ui.prometheus.enabled": "true",
    "spark.metrics.conf.*.sink.prometheusServlet.class": "org.apache.spark.metrics.sink.PrometheusServlet",
    "spark.metrics.conf.*.sink.prometheusServlet.path": "/metrics/prometheus",
    "spark.metrics.conf.*.sink.jmx.class": "org.apache.spark.metrics.sink.JmxSink",
    "spark.metrics.conf.*.source.jvm.class": "org.apache.spark.metrics.source.JvmSource",
    "spark.metrics.conf.*.source.DAGSchedulerSource.class": "org.apache.spark.metrics.source.DAGSchedulerSource",
    "spark.metrics.conf.*.source.BlockManagerSource.class": "org.apache.spark.metrics.source.BlockManagerSource",
    "spark.metrics.conf.*.source.ExecutorSource.class": "org.apache.spark.metrics.source.ExecutorSource"
}