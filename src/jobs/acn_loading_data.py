import sys
import time
import importlib
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

def run_job(config_module):
    # 1. Load config động dựa trên tham số truyền vào
    cfg = importlib.import_module(config_module)

    # 2. Khởi tạo Spark với config từ file
    builder = SparkSession.builder.appName(cfg.APP_NAME)
    for key, value in cfg.SPARK_CONF.items():
        builder = builder.config(key, value)
    spark = builder.getOrCreate()

    # 3. Thực thi và đo đạc
    try:
        # Đo thời gian đọc
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

        # 4. In kết quả thực nghiệm
        print(f"\n[RESULT] Rows: {row_count} | Read: {read_time:.2f}s | Write: {write_time:.2f}s | Partitions: {num_partitions}")
        
        # (Tùy chọn) Ghi vào file CSV để sau này vẽ biểu đồ
        with open(cfg.METRICS_LOG, "a") as f:
            f.write(f"{cfg.APP_NAME},{row_count},{read_time},{write_time},{num_partitions}\n")

    finally:
        spark.stop()

if __name__ == "__main__":
    # Nhận tên file config từ command line (ví dụ: configs.exp1_31k)
    if len(sys.argv) < 2:
        print("Usage: spark-submit acn_loading_data.py <config_module_path>")
        sys.exit(1)
    
    run_job(sys.argv[1])