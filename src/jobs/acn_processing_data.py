import sys
import importlib
import json
import time
from datetime import datetime
from typing import Dict, Any

from src.utils.spark import create_spark
from src.utils.logger import get_logger
from src.transforms.acn import run_pipeline
from src.transforms.acn_validation import run_all_validations, log_validation_results


def save_processing_metadata(output_path: str, metadata: Dict[str, Any], logger):
    """Save processing metadata to JSON file"""
    try:
        # Try HDFS first, fallback to local
        metadata_path = f"{output_path}/_metadata.json"
        
        # For local filesystem
        import os
        local_path = metadata_path.replace("hdfs://localhost:9000", "")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        with open(local_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Metadata saved to {local_path}")
    except Exception as e:
        logger.warning(f"Could not save metadata: {e}")


def main(config_module: str):
    """Main processing job"""
    
    # Load config
    cfg = importlib.import_module(config_module)
    logger = get_logger(cfg.APP_NAME)
    
    # Start timing
    job_start_time = time.time()
    
    try:
        # Initialize Spark
        logger.info("=" * 70)
        logger.info(f"Starting {cfg.APP_NAME}")
        logger.info("=" * 70)
        
        spark = create_spark(cfg.APP_NAME, getattr(cfg, 'SPARK_CONF', {}))
        
        # Read input data
        logger.info(f"Reading data from {cfg.INPUT_PATH}")
        read_start = time.time()
        df = spark.read.parquet(cfg.INPUT_PATH)
        # 🔥 QUAN TRỌNG: đảm bảo có nhiều task
        df = df.repartition(cfg.NUM_PARTITIONS)
        df.cache()
        df.count()  # materialize cache
        original_count = df.count()
        original_cols = len(df.columns)
        read_time = time.time() - read_start
        
        logger.info(f"Loaded {original_count:,} rows, {original_cols} columns in {read_time:.2f}s")
        
        # Show sample
        logger.info("Sample data (first 2 rows):")
        df.show(2, truncate=False)
        
        # Save before state for validation
        df_before = df
        
        # Run transformation pipeline
        logger.info("Starting transformation pipeline...")
        transform_start = time.time()
        
        # Prepare config for pipeline
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
        
        # Run validations
        logger.info("Running validations...")
        validation_start = time.time()
        validation_results = run_all_validations(df_before, df_transformed, pipeline_config)
        validation_time = time.time() - validation_start
        
        # Log validation results
        critical_issues = log_validation_results(validation_results, logger)
        
        # Optional: fail job if critical issues found
        if critical_issues and getattr(cfg, 'FAIL_ON_VALIDATION_ERROR', False):
            raise ValueError(f"Validation failed with critical issues: {critical_issues}")
        
        # Repartition if needed
        if getattr(cfg, 'NUM_PARTITIONS', None):
            logger.info(f"Repartitioning to {cfg.NUM_PARTITIONS} partitions")
            df_transformed = df_transformed.repartition(cfg.NUM_PARTITIONS)
        
        # Write output
        logger.info(f"Writing to {cfg.OUTPUT_PATH}")
        write_start = time.time()
        df_transformed.write.mode("overwrite").option("maxRecordsPerFile", 500000).parquet(cfg.OUTPUT_PATH)
        logger.info("Holding job for metrics scraping...")
        # time.sleep(600)
        logger.info("Starting executor keep-alive jobs...")
        keep_executors_alive(spark)
        # giữ app sống để scrape metrics
        time.sleep(600)
        
        
        write_time = time.time() - write_start
        
        # Final counts
        final_count = df_transformed.count()
        final_cols = len(df_transformed.columns)
        
        # Calculate job metrics
        job_end_time = time.time()
        total_time = job_end_time - job_start_time
        
        # Prepare job summary
        job_summary = {
            "app_name": cfg.APP_NAME,
            "timestamp": datetime.now().isoformat(),
            "input_path": cfg.INPUT_PATH,
            "output_path": cfg.OUTPUT_PATH,
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
        
        # Save metadata
        save_processing_metadata(cfg.OUTPUT_PATH, job_summary, logger)
        
        # Log final summary
        logger.info("=" * 70)
        logger.info("JOB COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)
        logger.info(f"Rows: {original_count:,} → {final_count:,} (removed {original_count - final_count:,})")
        logger.info(f"Cols: {original_cols} → {final_cols}")
        logger.info(f"Retention rate: {job_summary['metrics']['pct_rows_retained']:.2f}%")
        logger.info(f"\nTiming breakdown:")
        logger.info(f"  - Read: {read_time:.2f}s")
        logger.info(f"  - Transform: {transform_time:.2f}s")
        logger.info(f"  - Validation: {validation_time:.2f}s")
        logger.info(f"  - Write: {write_time:.2f}s")
        logger.info(f"  - Total: {total_time:.2f}s")
        logger.info(f"\nProcessing rate: {original_count / total_time:.0f} rows/second")
        logger.info("=" * 70)
        
        # Optional: save metrics to CSV for tracking
        if hasattr(cfg, 'METRICS_LOG') and cfg.METRICS_LOG:
            import os
            os.makedirs(os.path.dirname(cfg.METRICS_LOG), exist_ok=True)
            
            import csv
            file_exists = os.path.exists(cfg.METRICS_LOG)
            with open(cfg.METRICS_LOG, 'a') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow([
                        'timestamp', 'app_name', 'original_rows', 'final_rows',
                        'read_time', 'transform_time', 'write_time', 'total_time',
                        'rows_per_second', 'validation_passed'
                    ])
                writer.writerow([
                    datetime.now().isoformat(),
                    cfg.APP_NAME,
                    original_count,
                    final_count,
                    f"{read_time:.2f}",
                    f"{transform_time:.2f}",
                    f"{write_time:.2f}",
                    f"{total_time:.2f}",
                    f"{original_count / total_time:.0f}",
                    len(critical_issues) == 0
                ])
            logger.info(f"Metrics logged to {cfg.METRICS_LOG}")
        
    except Exception as e:
        logger.error(f"Job failed: {str(e)}", exc_info=True)
        sys.exit(1)
    
    finally:
        if 'spark' in locals():
            time.sleep(600)
            spark.stop()
            logger.info("Spark session stopped")

from pyspark.sql.functions import rand
import threading

def keep_executors_alive(spark, interval=10):
    """
    Run dummy Spark jobs to keep executors alive and generate metrics
    """
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
    
    
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: spark-submit src/jobs/acn_processing_data.py <config_module>")
        print("Example: spark-submit src/jobs/acn_processing_data.py configs.config_caltech")
        sys.exit(1)
    
    main(sys.argv[1])
