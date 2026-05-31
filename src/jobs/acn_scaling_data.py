# src/jobs/acn_scaling_data.py
import sys
import time
import os
import csv
import json
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import rand, lit, explode, sequence, col, concat, avg, stddev

# Thêm PROJECT_ROOT vào sys.path để gọi được file config nằm ở thư mục khác
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from configs import config_scaling as cfg


def save_metrics(cfg, metrics):
    """
    Lưu trữ metrics đồng bộ theo cấu trúc của acn_loading_data:
    - results/scaling/raw_metrics/*.json
    - results/scaling/scaling_metrics.csv
    """
    result_dir = cfg.RESULT_DIR
    raw_dir = os.path.join(result_dir, "raw_metrics")
    os.makedirs(raw_dir, exist_ok=True)

    # 1. Lưu file JSON chi tiết cấu hình và kết quả thực nghiệm
    json_file = os.path.join(raw_dir, f"{cfg.APP_NAME}_SF_{metrics['scale_factor']}.json")
    with open(json_file, "w") as f:
        json.dump(metrics, f, indent=4)

    # 2. Lưu file CSV tổng hợp để phục vụ thống kê, so sánh trong luận văn
    csv_file = os.path.join(result_dir, f"{cfg.APP_NAME}_SF_metrics.csv")
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "experiment",
                "scale_factor",
                "approx_size_mb",
                "input_rows",
                "output_rows",
                "read_time_sec",
                "scaling_time_sec",
                "write_time_sec",
                "throughput_scaling_rows_per_sec",
                "input_partitions",
                "output_partitions"
            ])

        writer.writerow([
            metrics["timestamp"],
            metrics["experiment"],
            metrics["scale_factor"],
            metrics["approx_size_mb"],
            metrics["input_rows"],
            metrics["output_rows"],
            metrics["read_time_sec"],
            metrics["scaling_time_sec"],
            metrics["write_time_sec"],
            metrics["throughput_scaling_rows_per_sec"],
            metrics["input_partitions"],
            metrics["output_partitions"]
        ])


def init_spark_session(app_name: str, conf_dict: dict) -> SparkSession:
    """Khởi tạo và cấu hình Spark Session từ file config."""
    builder = SparkSession.builder.appName(app_name)
    for key, value in conf_dict.items():
        builder = builder.config(key, value)
    return builder.getOrCreate()


def load_input_data(spark: SparkSession, input_path: str) -> tuple:
    """Đọc dữ liệu nguồn (HDFS/Local) và đo lường thời gian đọc."""
    start_time = time.time()
    
    if "parquet" in input_path.lower():
        df = spark.read.parquet(input_path)
    else:
        df = spark.read.json(input_path)
        
    read_time = time.time() - start_time
    return df, read_time


def apply_replication_and_perturbation(df, scale_factor: int) -> tuple:
    """
    Thực hiện nhân bản dữ liệu theo Scale Factor và áp dụng các kỹ thuật nhiễu toán học:
    - ID Uniqueness: Tạo khóa duy nhất tránh duplicate key.
    - Numeric Variation: Tạo biến động ±10% cho điện năng tiêu thụ.
    - Noise Generation: Thêm feature nhiễu phục vụ ML benchmarking.
    """
    start_time = time.time()

    # 1. Volumetric Scaling (Nhân bản dòng thông qua explode sequence)
    df_scaled = df.withColumn(
        "batch_id", 
        explode(sequence(lit(0), lit(scale_factor - 1)))
    )

    # 2. Xử lý ID duy nhất tránh trùng lặp thực thể khi nhân bản
    df_scaled = df_scaled.withColumn(
        "userID", concat(col("userID"), lit("_"), col("batch_id"))
    ).withColumn(
        "sessionID", concat(col("sessionID"), lit("_"), col("batch_id"))
    )

    # 3. Áp dụng biến động số học (±10% kWhDelivered) nhằm bảo toàn dạng phân phối
    df_scaled = df_scaled.withColumn(
        "kWhDelivered", 
        col("kWhDelivered") * (1 + (rand() - 0.5) * 0.2)
    )

    # 4. Thêm tính năng nhiễu hệ thống (noise column)
    df_scaled = df_scaled.withColumn("noise", rand())

    # Ép Spark tính toán ngẫu nhiên đồng nhất và lưu tạm thời trên RAM
    df_scaled.cache()
    output_rows = df_big_count = df_scaled.count()
    
    scaling_time = time.time() - start_time
    return df_scaled, output_rows, scaling_time


def calculate_distribution_validation(df_source, df_target, scale_factor: int):
    """
    Xác thực phân phối toán học phục vụ lập luận thesis:
    Chứng minh dữ liệu sinh ra giữ nguyên Mean và StdDev của cột core 'kWhDelivered'.
    """
    stats_src = df_source.select(avg("kWhDelivered").alias("mean_kwh"), stddev("kWhDelivered").alias("std_kwh")).collect()[0]
    stats_tgt = df_target.select(avg("kWhDelivered").alias("mean_kwh"), stddev("kWhDelivered").alias("std_kwh")).collect()[0]
    
    with open(cfg.SCALING_VALIDATION_CSV, "a") as f:
        f.write(
            f"{scale_factor},{stats_src['mean_kwh']:.4f},{stats_tgt['mean_kwh']:.4f},"
            f"{stats_src['std_kwh']:.4f},{stats_tgt['std_kwh']:.4f}\n"
        )


def write_output_data(df, output_path: str, partitions: int) -> float:
    """Phân rã lại partition tối ưu hóa IO và ghi dữ liệu nén Snappy xuống HDFS."""
    start_time = time.time()
    
    # Tăng gấp đôi số lượng partition cho file đầu ra có dung lượng lớn
    df.repartition(partitions * 2) \
      .write \
      .mode("overwrite") \
      .option("compression", "snappy") \
      .parquet(output_path)
      
    return time.time() - start_time


def log_experiment_metrics(input_rows, output_rows, scale_factor, read_time, scaling_time, write_time, in_parts, out_parts):
    """In báo cáo hiệu năng trực quan ra console màn hình terminal."""
    rows_per_sec = output_rows / scaling_time if scaling_time > 0 else 0
    
    report = f"""
==================================================================
[SCALING JOB SUCCESS] - Scale Factor: {scale_factor}
==================================================================
Input Records      : {input_rows:,} (Partitions: {in_parts})
Output Records     : {output_rows:,} (Partitions: {out_parts})
------------------------------------------------------------------
Read Performance   : {read_time:.2f} seconds
Scaling Throughput : {scaling_time:.2f} seconds ({rows_per_sec:.2f} rec/sec)
Write Performance  : {write_time:.2f} seconds
Total Core Time    : {read_time + scaling_time + write_time:.2f} seconds
==================================================================
"""
    print(report)


def main():
    # Phân tích tham số CLI, nếu thiếu sẽ tự động fallback về cấu hình mặc định trong config
    scale_factor = int(sys.argv[1]) if len(sys.argv) > 1 else cfg.DEFAULT_SCALE_FACTOR
    partitions = int(sys.argv[2]) if len(sys.argv) > 2 else cfg.DEFAULT_PARTITIONS
    
    # Tạo dynamic output path và nhãn dung lượng ước tính
    size_label = "1gb" if scale_factor >= 50 else "500mb"
    approx_size_mb = 1024 if scale_factor >= 50 else 500
    target_output_path = f"{cfg.OUTPUT_BASE_PATH}/{size_label}"

    print(f"[*] Starting Scaling Job with Scale Factor: {scale_factor} -> Target: {target_output_path}")

    spark = init_spark_session(cfg.APP_NAME, cfg.SPARK_CONF)

    try:
        # 1. Pipeline Execution: Load
        df_src, read_time = load_input_data(spark, cfg.INPUT_PATH)
        input_rows = df_src.count()
        input_partitions = df_src.rdd.getNumPartitions()

        # Tổ chức lại partition ban đầu và đưa vào cache để tăng tốc độ xử lý
        df_src = df_src.repartition(partitions).cache()

        # 2. Pipeline Execution: Scale & Perturb
        df_scaled, output_rows, scaling_time = apply_replication_and_perturbation(df_src, scale_factor)
        
        # Thống kê toán học kiểm định dữ liệu nhân bản phục vụ nghiên cứu luận văn
        calculate_distribution_validation(df_src, df_scaled, scale_factor)

        # 3. Pipeline Execution: Write
        write_time = write_output_data(df_scaled, target_output_path, partitions)
        output_partitions = partitions * 2

        # 4. Tính toán Throughput cụ thể cho tác vụ Scaling
        throughput_scaling = output_rows / scaling_time if scaling_time > 0 else 0

        # 5. Đóng gói cấu trúc Log Metrics (Đồng bộ chuẩn hóa với acn_loading_data.py)
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "experiment": f"{cfg.APP_NAME}_SF_{scale_factor}",
            "scale_factor": scale_factor,
            "approx_size_mb": approx_size_mb,
            "input_rows": input_rows,
            "output_rows": output_rows,
            "read_time_sec": round(read_time, 3),
            "scaling_time_sec": round(scaling_time, 3),
            "write_time_sec": round(write_time, 3),
            "throughput_scaling_rows_per_sec": round(throughput_scaling, 2),
            "input_partitions": input_partitions,
            "output_partitions": output_partitions,
            "input_path": cfg.INPUT_PATH,
            "output_path": target_output_path,
            "spark_conf": cfg.SPARK_CONF
        }
        
        # Gọi hàm lưu song song CSV và JSON
        save_metrics(cfg, metrics)

        # Hiển thị kết quả ra Console terminal
        log_experiment_metrics(
            input_rows, output_rows, scale_factor, 
            read_time, scaling_time, write_time, 
            input_partitions, output_partitions
        )
        print("[INFO] Metrics saved successfully in JSON and CSV formats.")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()