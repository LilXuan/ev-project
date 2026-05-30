# src/jobs/acn_processing_data.py
import sys
import os
import importlib
import json
import time
import csv
import threading
from datetime import datetime
from typing import Dict, Any
from pyspark.sql.functions import rand

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.utils.spark import create_spark
from src.utils.logger import get_logger
from src.transforms.acn import run_pipeline
from src.transforms.acn_validation import run_all_validations, log_validation_results


def save_metrics(cfg, metrics: Dict[str, Any], logger):
    """
    Lưu trữ metrics đồng bộ theo cấu trúc của loading và scaling:
    - results/processing/raw_metrics/*.json
    - results/processing/processing_metrics.csv
    """
    try:
        result_dir = cfg.RESULT_DIR
        raw_dir = os.path.join(result_dir, "raw_metrics")
        os.makedirs(raw_dir, exist_ok=True)

        # 1. Lưu file JSON chi tiết cấu hình và kết quả validation
        json_file = os.path.join(raw_dir, f"{cfg.APP_NAME}.json")
        with open(json_file, "w") as f:
            json.dump(metrics, f, indent=4, default=str)
        logger.info(f"Detailed processing metadata successfully saved to JSON: {json_file}")

        # 2. Lưu file CSV tổng hợp phục vụ so sánh và vẽ biểu đồ hiệu năng pipeline ML
        csv_file = cfg.METRICS_LOG
        file_exists = os.path.isfile(csv_file)

        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow([
                    "timestamp",
                    "experiment",
                    "dataset_size_mb",
                    "original_rows",
                    "final_rows",
                    "rows_removed",
                    "pct_rows_retained",
                    "original_columns",
                    "final_columns",
                    "read_time_sec",
                    "transform_time_sec",
                    "validation_time_sec",
                    "write_time_sec",
                    "total_time_sec",
                    "throughput_rows_per_sec",
                    "validation_passed"
                ])

            writer.writerow([
                metrics["timestamp"],
                metrics["experiment"],
                metrics["dataset_size_mb"],
                metrics["metrics"]["original_rows"],
                metrics["metrics"]["final_rows"],
                metrics["metrics"]["rows_removed"],
                round(metrics["metrics"]["pct_rows_retained"], 2),
                metrics["metrics"]["original_columns"],
                metrics["metrics"]["final_columns"],
                round(metrics["timing"]["read_time_seconds"], 3),
                round(metrics["timing"]["transform_time_seconds"], 3),
                round(metrics["timing"]["validation_time_seconds"], 3),
                round(metrics["timing"]["write_time_seconds"], 3),
                round(metrics["timing"]["total_time_seconds"], 3),
                round(metrics["throughput_rows_per_second"], 2),
                metrics["validation_passed"]
            ])
        logger.info(f"Summary metrics successfully appended to CSV: {csv_file}")

    except Exception as e:
        logger.error(f"Failed to save metrics under unified structure: {str(e)}", exc_info=True)


def keep_executors_alive(spark, interval=10):
    """Run dummy Spark jobs to keep executors alive and generate metrics"""
    def run():
        while True:
            try:
                spark.range(1000000) \
                    .repartition(8) \
                    .select(rand()) \
                    .count()
                time.sleep(interval)
            except Exception:
                break

    t = threading.Thread(target=run, daemon=True)
    t.start()


def main(config_module: str):
    """Main processing job"""
    # Load config động dựa trên tham số CLI truyền vào
    cfg = importlib.import_module(config_module)
    logger = get_logger(cfg.APP_NAME)
    
    # Kích hoạt đo lường thời gian toàn cục của Job
    job_start_time = time.time()
    
    try:
        logger.info("=" * 70)
        logger.info(f"Starting Processing Job: {cfg.APP_NAME}")
        logger.info("=" * 70)
        
        # Khởi tạo Spark Session
        spark = create_spark(cfg.APP_NAME, getattr(cfg, 'SPARK_CONF', {}))
        
        # ==========================================
        # GIAI ĐOẠN 1: ĐỌC DỮ LIỆU ĐẦU VÀO (PARQUET)
        # ==========================================
        logger.info(f"Reading parquet data from {cfg.INPUT_PATH}")
        read_start = time.time()
        df = spark.read.parquet(cfg.INPUT_PATH)
        
        # Đồng bộ repartition tối ưu hóa tải trọng xử lý song song
        df = df.repartition(cfg.NUM_PARTITIONS)
        df.cache()
        df.count()  # Thực hiện Action để load cứng dữ liệu vào cache RAM
        
        original_count = df.count()
        original_cols = len(df.columns)
        read_time = time.time() - read_start
        
        logger.info(f"Loaded {original_count:,} rows, {original_cols} columns in {read_time:.2f}s")
        
        logger.info("Sample data (first 2 rows):")
        df.show(2, truncate=False)
        
        df_before = df
        
        # ==========================================
        # GIAI ĐOẠN 2: THỰC THI PIPELINE BIẾN ĐỔI (FEATURE ENGINEERING)
        # ==========================================
        logger.info("Starting transformation pipeline...")
        transform_start = time.time()
        
        pipeline_config = {
            'features': getattr(cfg, 'FEATURES', None),
            'critical_cols': getattr(cfg, 'CRITICAL_COLS', None),
            'outlier_cols': getattr(cfg, 'OUTLIER_COLS', ["duration", "charging_duration", "kWhDelivered"]),
            'lags': getattr(cfg, 'LAGS', (1, 2, 3)),
            'windows': getattr(cfg, 'WINDOWS', (3, 5))
        }
        
        df_transformed, metadata = run_pipeline(df, pipeline_config)
        transform_time = time.time() - transform_start
        logger.info(f"Transformation completed in {transform_time:.2f}s")
        
        # ==========================================
        # GIAI ĐOẠN 3: XÁC THỰC TOÀN VẸN DỮ LIỆU (VALIDATION)
        # ==========================================
        logger.info("Running validation checks...")
        validation_start = time.time()
        validation_results = run_all_validations(df_before, df_transformed, pipeline_config)
        validation_time = time.time() - validation_start
        
        critical_issues = log_validation_results(validation_results, logger)
        
        if critical_issues and getattr(cfg, 'FAIL_ON_VALIDATION_ERROR', False):
            raise ValueError(f"Validation failed with critical issues: {critical_issues}")
        
        if getattr(cfg, 'NUM_PARTITIONS', None):
            logger.info(f"Repartitioning transformed data to {cfg.NUM_PARTITIONS} partitions")
            df_transformed = df_transformed.repartition(cfg.NUM_PARTITIONS)
        
        # ==========================================
        # GIAI ĐOẠN 4: GHI DỮ LIỆU ĐẦU RA (GOLD LAYER) & GIỮ APP ĐỂ SCRAPE
        # ==========================================
        logger.info(f"Writing output parquet to {cfg.OUTPUT_PATH}")
        write_start = time.time()
        df_transformed.write.mode("overwrite").option("maxRecordsPerFile", 500000).parquet(cfg.OUTPUT_PATH)
        
        logger.info("Starting executor keep-alive jobs...")
        keep_executors_alive(spark)
        
        logger.info("Holding job for 600 seconds to allow complete Prometheus/JMX metrics scraping...")
        time.sleep(600)  # Giữ session sống phục vụ hệ thống giám sát tải Prometheus
        
        write_time = time.time() - write_start
        
        final_count = df_transformed.count()
        final_cols = len(df_transformed.columns)
        
        job_end_time = time.time()
        total_time = job_end_time - job_start_time
        
        # ==========================================
        # GIAI ĐOẠN 5: ĐÓNG GÓI VÀ LƯU METRICS ĐỒNG BỘ
        # ==========================================
        job_summary = {
            "app_name": cfg.APP_NAME,
            "experiment": cfg.APP_NAME,
            "timestamp": datetime.now().isoformat(),
            "dataset_size_mb": getattr(cfg, 'DATASET_SIZE_MB', 1024),
            "input_path": cfg.INPUT_PATH,
            "output_path": cfg.OUTPUT_PATH,
            "validation_passed": len(critical_issues) == 0,
            "throughput_rows_per_second": round(original_count / total_time, 2),
            "metrics": {
                "original_rows": original_count,
                "final_rows": final_count,
                "rows_removed": original_count - final_count,
                "pct_rows_retained": (final_count / original_count) * 100 if original_count > 0 else 0,
                "original_columns": original_cols,
                "final_columns": final_cols,
                "columns_removed": original_cols - final_cols
            },
            "timing": {
                "read_time_seconds": read_time,
                "transform_time_seconds": transform_time,
                "validation_time_seconds": validation_time,
                "write_time_seconds": write_time,
                "total_time_seconds": total_time
            },
            "processing_metadata": metadata,
            "validation_results": validation_results,
            "spark_config": {
                "master": spark.sparkContext.master,
                "app_id": spark.sparkContext.applicationId,
                "default_parallelism": spark.sparkContext.defaultParallelism
            }
        }
        
        # Gọi hàm xử lý lưu trữ tập trung
        save_metrics(cfg, job_summary, logger)
        
        logger.info("=" * 70)
        logger.info("JOB PREPROCESSING COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"Job failed during execution: {str(e)}", exc_info=True)
        sys.exit(1)
        
    finally:
        if 'spark' in locals():
            spark.stop()
            logger.info("Spark session cleanly stopped.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: spark-submit src/jobs/acn_processing_data.py <config_module>")
        sys.exit(1)
        
    main(sys.argv[1])