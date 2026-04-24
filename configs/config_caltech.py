# config_caltech.py
import os
from pathlib import Path

APP_NAME = "ACN_Caltech_Preprocessing_Local"

# Input: Parquet from silver layer (local path or HDFS)
INPUT_PATH = "hdfs://localhost:9000/ev-project/data/silver/ev_sessions/caltech/*"

# Output: Processed data ready for analysis
OUTPUT_PATH = "hdfs://localhost:9000/ev-project/data/gold/ev_sessions/caltech_processed"

# Training output paths
TRAINING_OUTPUT_PATH = "hdfs://localhost:9000/ev-project/data/gold/ev_sessions/caltech_training"
MODEL_OUTPUT_PATH = "hdfs://localhost:9000/ev-project/models/gbt_caltech"
PREDICTIONS_OUTPUT_PATH = "hdfs://localhost:9000/ev-project/predictions/gbt_caltech"

# Log files - sử dụng thư mục trong project
PROJECT_ROOT = Path(__file__).parent.parent  # Đi lên 2 cấp từ configs/
METRICS_DIR = PROJECT_ROOT / "metrics"
METRICS_LOG = str(METRICS_DIR / "preprocessing_metrics.csv")
TRAINING_METRICS_LOG = str(METRICS_DIR / "training_metrics.csv")
RESEARCH_LOG = str(METRICS_DIR / "preprocessing_stats.json")

# Tạo thư mục metrics nếu chưa có
METRICS_DIR.mkdir(exist_ok=True)

# Optional: repartition for optimal write
# Processing configuration
NUM_PARTITIONS = 4  # For local testing
FAIL_ON_VALIDATION_ERROR = False

# Feature configuration - QUAN TRỌNG: Định nghĩa rõ ràng
FEATURES = [
    'hour', 'day_of_week', 'month', 'season',
    'duration', 'charging_duration', 'charging_duration_log',
    'hour_sin', 'hour_cos', 'day_of_year', 'week_of_year', 'is_holiday',
    'lag_1_log', 'lag_2_log', 'lag_3_log',
    'rolling_mean_3_log', 'rolling_mean_5_log'
]

CRITICAL_COLS = [
    'kWhDelivered', 'duration', 'charging_duration_log'
]

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

# Optional: Use subset to match paper's size (for comparison only)
USE_PAPER_SUBSET = False  # Set to True to test with 14,496 sessions
PAPER_SUBSET_SIZE = 14496

SPARK_CONF = {
    "spark.executor.memory": "4g",  # Tăng lên cho training
    "spark.executor.cores": "2",
    "spark.sql.shuffle.partitions": "20",  # Tăng cho training
    "spark.driver.memory": "2g"
}