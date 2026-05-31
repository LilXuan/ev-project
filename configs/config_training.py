# configs/config_training.py
import os
from pathlib import Path
from configs import config_processing as proc_cfg

APP_NAME = "ACN_Caltech_Training_1GB"

# Kế thừa thông tin kích thước dữ liệu từ giai đoạn tiền xử lý để đồng bộ
SIZE_LABEL = proc_cfg.SIZE_LABEL

# Đường dẫn đọc dữ liệu đã xử lý (Gold Layer) và xuất kết quả Model
INPUT_PATH = f"hdfs://localhost:9000/ev-project/data/silver/ev_sessions/1gb"
TRAINING_OUTPUT_PATH = f"hdfs://localhost:9000/ev-project/data/gold/ev_sessions/1gb"
MODEL_OUTPUT_PATH = f"hdfs://localhost:9000/ev-project/models/gbt_caltech/1gb"
PREDICTIONS_OUTPUT_PATH = f"hdfs://localhost:9000/ev-project/predictions/gbt_caltech/1gb"

# Quản lý đường dẫn kết quả Training nội bộ dự án
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_RESULT_DIR = os.path.join(str(PROJECT_ROOT), "results", "training")
os.makedirs(TRAIN_RESULT_DIR, exist_ok=True)

TRAINING_METRICS_LOG = os.path.join(TRAIN_RESULT_DIR, f"{APP_NAME}_training_metrics.csv")
CHECKPOINT_PATH = "/tmp/acn_checkpoint"

# Siêu tham số (Hyperparameters) cấu hình cho GBT Regressor Pipeline
TRAIN_CONFIG = {
    "train_fraction": 0.8,
    "seed": 42,
    "max_depth": 10,
    "max_iter": 100,
    "feature_cols": proc_cfg.FEATURES, # Tham chiếu trực tiếp đến danh sách features ổn định
    "target_col": "kWhDelivered_log"
}

# Tham số cấu hình kiểm định theo bài báo nghiên cứu gốc
USE_PAPER_SUBSET = False
PAPER_SUBSET_SIZE = 14496

# Cấu hình tài nguyên Spark Cluster - Được tăng cường RAM/Core để chịu tải tính toán ML lớn
SPARK_CONF = {
    "spark.executor.memory": "2g",        # Tăng cường bộ nhớ đệm cho quá trình huấn luyện cây quyết định
    "spark.executor.cores": "2",
    "spark.default.parallelism": "8",
    "spark.sql.shuffle.partitions": "8",
    "spark.driver.memory": "2g",          # Tăng bộ nhớ driver để thu thập kết quả đánh giá (Evaluation)
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