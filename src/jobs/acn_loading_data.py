import sys
import time
import importlib
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

import os
import csv
import json

from datetime import datetime


def save_metrics(cfg, metrics):
    """
    Save metrics to:
    results/loading/raw_metrics/*.json
    results/loading/loading_metrics.csv
    """

    result_dir = cfg.RESULT_DIR
    raw_dir = os.path.join(result_dir, "raw_metrics")

    os.makedirs(raw_dir, exist_ok=True)

    # save json file
    json_file = os.path.join(raw_dir,f"{cfg.APP_NAME}.json")
    with open(json_file, "w") as f:
        json.dump(metrics, f, indent=4)


    # save csv file
    csv_file = os.path.join(result_dir,"loading_metrics.csv")
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, "a", newline="") as f:

        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "experiment",
                "dataset_size_mb",
                "rows",
                "read_time_sec",
                "write_time_sec",
                "throughput_read_rows_per_sec",
                "throughput_write_rows_per_sec",
                "num_partitions"
            ])

        writer.writerow([
            metrics["timestamp"],
            metrics["experiment"],
            metrics["dataset_size_mb"],
            metrics["rows"],
            metrics["read_time_sec"],
            metrics["write_time_sec"],
            metrics["throughput_read_rows_per_sec"],
            metrics["throughput_write_rows_per_sec"],
            metrics["num_partitions"]
        ])

def create_spark_session(cfg):
    builder = SparkSession.builder.appName(cfg.APP_NAME)
    for key, value in cfg.SPARK_CONF.items():
        builder = builder.config(key, value)
    return builder.getOrCreate()


def run_job(config_module):
    # 1. Load config động dựa trên tham số truyền vào
    cfg = importlib.import_module(config_module)

    # 2. Khởi tạo Spark với config từ file
    spark = create_spark_session(cfg)
    
    # 3. Thực thi và đo đạc
    try:
        # Đo thời gian đọc từ bronze trên hdfs
        start_read = time.time()
        df = spark.read.json(cfg.INPUT_PATH)
        row_count = df.count() 
        read_time = time.time() - start_read

        # Lấy thông số vật lý
        num_partitions = df.rdd.getNumPartitions()

        # Đo thời gian ghi (Parquet)
        start_write = time.time()
        df.write.mode("overwrite").parquet(cfg.OUTPUT_PATH)
        write_time = time.time() - start_write


        # ĐO Throughput
        throughput_read = (
            row_count / read_time
            if read_time > 0 else 0
        )

        throughput_write = (
            row_count / write_time
            if write_time > 0 else 0
        )

        # Save metrics 
        metrics = {
            "timestamp":
                datetime.now().isoformat(),
            "experiment":
                cfg.APP_NAME,
            "dataset_size_mb":
                cfg.DATASET_SIZE_MB,
            "rows":
                row_count,
            "read_time_sec":
                round(read_time, 3),
            "write_time_sec":
                round(write_time, 3),
            "throughput_read_rows_per_sec":
                round(
                    throughput_read,
                    2
                ),
            "throughput_write_rows_per_sec":
                round(
                    throughput_write,
                    2
                ),
            "num_partitions":
                num_partitions,
            "input_path":
                cfg.INPUT_PATH,
            "output_path":
                cfg.OUTPUT_PATH,
            "spark_conf":
                cfg.SPARK_CONF
        }
        save_metrics(cfg, metrics)

        # Console output
        print(
            f"[RESULT] "
            f"Rows={row_count:,} | "
            f"Read={read_time:.2f}s | "
            f"Write={write_time:.2f}s | "
            f"Partitions={num_partitions}"
        )

        print(
            f"[THROUGHPUT] "
            f"Read={throughput_read:.2f} rows/s | "
            f"Write={throughput_write:.2f} rows/s"
        )

        print(
            "[INFO] Metrics saved successfully."
        )
        
    finally:
        spark.stop()

if __name__ == "__main__":
    # Nhận tên file config từ command line (ví dụ: configs.exp1_31k)
    if len(sys.argv) < 2:
        print(
            "Usage:\n"
            "spark-submit acn_loading_data.py"
            "<config_module_path>"
        )
        sys.exit(1)
    
    run_job(sys.argv[1])