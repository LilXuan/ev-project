#!/usr/bin/env python3
"""
ACN Training Model Job - GBT Regressor for EV Charging Prediction
Simple orchestrator that calls training and validation modules
"""

import sys
import os
import importlib
import json
import csv
from datetime import datetime

# Đảm bảo hệ thống nạp đúng đường dẫn thư mục gốc dự án
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.utils.spark import create_spark
from src.utils.logger import get_logger
from src.transforms.acn_training import run_training_pipeline
from src.transforms.acn_train_validation import (
    calculate_metrics,
    log_metrics,
    compare_with_paper,
    validate_predictions
)


def save_training_metadata(cfg, metadata: dict, logger):
    """
    Lưu trữ chi tiết metadata cấu hình huấn luyện dưới dạng tệp JSON:
    - results/training/raw_metrics/*.json
    """
    try:
        # Gom tập trung tệp tin vào thư mục kết quả huấn luyện nội bộ
        result_dir = cfg.TRAIN_RESULT_DIR
        raw_dir = os.path.join(result_dir, "raw_metrics")
        os.makedirs(raw_dir, exist_ok=True)
        
        json_file = os.path.join(raw_dir, f"{cfg.APP_NAME}.json")
        
        with open(json_file, 'w') as f:
            json.dump(metadata, f, indent=4, default=str)
        
        logger.info(f"Detailed training metadata successfully saved to JSON: {json_file}")
    except Exception as e:
        logger.warning(f"Could not save training metadata to primary location: {e}")
        # Cơ chế Fallback an toàn nếu hệ thống phân quyền cục bộ bị lỗi
        try:
            fallback_path = "training_metadata_fallback.json"
            with open(fallback_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            logger.info(f"Saved metadata to fallback location: {fallback_path}")
        except:
            logger.error("Could not save metadata even to fallback location")


def save_training_metrics_csv(
    train_metrics: dict,
    test_metrics: dict,
    validation_results: dict,
    training_time: float,
    dataset_size: int,
    config: dict,
    metrics_log_path: str,
    logger
):
    """
    Lưu trữ tổng hợp chỉ số sai số (Metrics) của 3 tập phân tách (Train, Test, Validation)
    đồng bộ theo cấu trúc bảng CSV của hệ thống:
    - results/training/training_metrics.csv
    """
    timestamp = datetime.now().isoformat()
    paper_metrics = {'mae_kwh': 2.6697, 'rmse_kwh': 3.9451}
    
    records = []
    
    # 1. Đóng gói bản ghi dữ liệu cho tập huấn luyện (Train Set)
    records.append({
        'timestamp': timestamp,
        'experiment': config.get('experiment_name', 'GBT_EV_Prediction'),
        'split': 'Train',
        'dataset_size': dataset_size,
        'train_fraction': config.get('train_fraction', 0.8),
        'seed': config.get('seed', 42),
        'max_depth': config.get('max_depth', 10),
        'max_iter': config.get('max_iter', 100),
        'training_time_sec': round(training_time, 2),
        'rows': train_metrics.get('rows', 0),
        'mae_log': round(train_metrics.get('mae_log', 0), 4) if train_metrics.get('mae_log') is not None else None,
        'rmse_log': round(train_metrics.get('rmse_log', 0), 4) if train_metrics.get('rmse_log') is not None else None,
        'r2_log': round(train_metrics.get('r2_log', 0), 4) if train_metrics.get('r2_log') is not None else None,
        'mae_kwh': round(train_metrics.get('mae_kwh', 0), 4) if train_metrics.get('mae_kwh') is not None else None,
        'rmse_kwh': round(train_metrics.get('rmse_kwh', 0), 4) if train_metrics.get('rmse_kwh') is not None else None,
        'r2_kwh': round(train_metrics.get('r2_kwh', 0), 4) if train_metrics.get('r2_kwh') is not None else None,
        'diff_mae_kwh': round(train_metrics.get('mae_kwh', 0) - paper_metrics['mae_kwh'], 4) if train_metrics.get('mae_kwh') is not None else None,
        'diff_rmse_kwh': round(train_metrics.get('rmse_kwh', 0) - paper_metrics['rmse_kwh'], 4) if train_metrics.get('rmse_kwh') is not None else None,
        'mean_pct_error': None,
        'outlier_pct': None
    })
    
    # 2. Đóng gói bản ghi dữ liệu cho tập kiểm thử (Test Set)
    records.append({
        'timestamp': timestamp,
        'experiment': config.get('experiment_name', 'GBT_EV_Prediction'),
        'split': 'Test',
        'dataset_size': dataset_size,
        'train_fraction': config.get('train_fraction', 0.8),
        'seed': config.get('seed', 42),
        'max_depth': config.get('max_depth', 10),
        'max_iter': config.get('max_iter', 100),
        'training_time_sec': round(training_time, 2),
        'rows': test_metrics.get('rows', 0),
        'mae_log': round(test_metrics.get('mae_log', 0), 4) if test_metrics.get('mae_log') is not None else None,
        'rmse_log': round(test_metrics.get('rmse_log', 0), 4) if test_metrics.get('rmse_log') is not None else None,
        'r2_log': round(test_metrics.get('r2_log', 0), 4) if test_metrics.get('r2_log') is not None else None,
        'mae_kwh': round(test_metrics.get('mae_kwh', 0), 4) if test_metrics.get('mae_kwh') is not None else None,
        'rmse_kwh': round(test_metrics.get('rmse_kwh', 0), 4) if test_metrics.get('rmse_kwh') is not None else None,
        'r2_kwh': round(test_metrics.get('r2_kwh', 0), 4) if test_metrics.get('r2_kwh') is not None else None,
        'diff_mae_kwh': round(test_metrics.get('mae_kwh', 0) - paper_metrics['mae_kwh'], 4) if test_metrics.get('mae_kwh') is not None else None,
        'diff_rmse_kwh': round(test_metrics.get('rmse_kwh', 0) - paper_metrics['rmse_kwh'], 4) if test_metrics.get('rmse_kwh') is not None else None,
        'mean_pct_error': None,
        'outlier_pct': None
    })
    
    # 3. Đóng gói bản ghi kiểm định (Validation Set - Đối chiếu toán học nâng cao)
    if validation_results:
        records.append({
            'timestamp': timestamp,
            'experiment': config.get('experiment_name', 'GBT_EV_Prediction'),
            'split': 'Validation',
            'dataset_size': dataset_size,
            'train_fraction': config.get('train_fraction', 0.8),
            'seed': config.get('seed', 42),
            'max_depth': config.get('max_depth', 10),
            'max_iter': config.get('max_iter', 100),
            'training_time_sec': round(training_time, 2),
            'rows': validation_results.get('total_rows', 0),
            'mae_log': None,
            'rmse_log': None,
            'r2_log': None,
            'mae_kwh': round(validation_results.get('mean_absolute_error', 0), 4),
            'rmse_kwh': round(validation_results.get('rmse_error', 0), 4),
            'r2_kwh': None,
            'diff_mae_kwh': round(validation_results.get('mean_absolute_error', 0) - paper_metrics['mae_kwh'], 4),
            'diff_rmse_kwh': round(validation_results.get('rmse_error', 0) - paper_metrics['rmse_kwh'], 4),
            'mean_pct_error': round(validation_results.get('mean_percentage_error', 0), 2),
            'outlier_pct': round(validation_results.get('outlier_percentage', 0), 2)
        })
    
    # Ghi xuất và lưu trữ dữ liệu CSV xuống kết quả tập trung
    os.makedirs(os.path.dirname(metrics_log_path), exist_ok=True)
    file_exists = os.path.exists(metrics_log_path)
    
    with open(metrics_log_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        if not file_exists:
            writer.writeheader()
        for record in records:
            writer.writerow(record)
    
    logger.info(f"Summary training metrics successfully appended to CSV: {metrics_log_path}")


def save_model_predictions(predictions_df, output_path: str, logger, mode: str = "overwrite"):
    """Ghi xuất kết quả dự đoán của mô hình (Predictions) xuống Gold Layer dạng Parquet"""
    try:
        logger.info(f"Saving predictions to {output_path}")
        
        if hasattr(predictions_df, 'repartition'):
            predictions_df = predictions_df.repartition(4)
        
        predictions_df.write.mode(mode).parquet(output_path)
        logger.info(f"Predictions saved successfully to {output_path}")
        
        count = predictions_df.count()
        logger.info(f"Total rows saved in prediction set: {count:,}")
    except Exception as e:
        logger.error(f"Failed to save predictions: {e}")
        raise


def main(config_module: str):
    """Main training job - simple orchestrator"""
    # Load cấu hình động từ file cấu hình huấn luyện riêng biệt (config_training)
    cfg = importlib.import_module(config_module)
    logger = get_logger(f"{cfg.APP_NAME}_Training")
    
    spark = None
    try:
        logger.info("=" * 70)
        logger.info(f"Starting Training Pipeline: {cfg.APP_NAME}")
        logger.info("=" * 70)
        
        # Khởi tạo hoặc nạp Spark Session
        spark = create_spark(f"{cfg.APP_NAME}_Training", getattr(cfg, 'SPARK_CONF', {}))
        
        # Đọc dữ liệu đầu vào đã qua tiền xử lý hoàn chỉnh (Gold Layer từ HDFS)
        logger.info(f"Loading gold layer preprocessed data from {cfg.INPUT_PATH}")
        df = spark.read.parquet(cfg.INPUT_PATH)
        total_rows = df.count()
        logger.info(f"Loaded {total_rows:,} records, {len(df.columns)} feature columns")
        
        # Mô phỏng tập dữ liệu giới hạn theo quy mô phân tích của bài báo gốc
        if getattr(cfg, 'USE_PAPER_SUBSET', False):
            target_count = getattr(cfg, 'PAPER_SUBSET_SIZE', 14496)
            fraction = target_count / total_rows
            logger.info(f"Subsetting to paper's size match: {target_count:,} sessions")
            df = df.sample(fraction=fraction, seed=cfg.TRAIN_CONFIG.get('seed', 42))
            df = df.cache()
            logger.info(f"Actual rows available after subset constraint: {df.count():,}")
        
        # === MÔ ĐUN HẤN LUYỆN MÔ HÌNH (Spark ML Pipeline) ===
        training_results = run_training_pipeline(
            df=df,
            feature_cols=cfg.TRAIN_CONFIG.get('feature_cols'),
            target_col=cfg.TRAIN_CONFIG.get('target_col', 'kWhDelivered_log'),
            train_fraction=cfg.TRAIN_CONFIG.get('train_fraction', 0.8),
            seed=cfg.TRAIN_CONFIG.get('seed', 42),
            max_depth=cfg.TRAIN_CONFIG.get('max_depth', 10),
            max_iter=cfg.TRAIN_CONFIG.get('max_iter', 100),
            use_scaler=True,
            cache_data=True,
            return_predictions=True
        )
        
        # === MÔ ĐUN ĐÁNH GIÁ VÀ TRÍCH XUẤT SAI SỐ (Validation & Evaluation) ===
        train_metrics = calculate_metrics(
            training_results['train_predictions'],
            target_col=cfg.TRAIN_CONFIG.get('target_col', 'kWhDelivered_log'),
            prediction_col="prediction"
        )
        
        test_metrics = calculate_metrics(
            training_results['test_predictions'],
            target_col=cfg.TRAIN_CONFIG.get('target_col', 'kWhDelivered_log'),
            prediction_col="prediction"
        )
        
        # Ghi vết chi tiết sai số ra terminal console log
        log_metrics(train_metrics, "TRAIN")
        log_metrics(test_metrics, "TEST")
        
        # Trích xuất bảng đối chiếu mô hình cục bộ với thuật toán học máy nâng cao của bài báo nghiên cứu
        from src.transforms.acn_train_validation import print_comparison_table, compare_with_paper_full
        print_comparison_table(train_metrics, test_metrics, model_name="GBT")
        compare_with_paper_full(test_metrics)
        
        differences = compare_with_paper(test_metrics)
        
        validation_results = validate_predictions(
            training_results['test_predictions'],
            threshold_kwh=getattr(cfg, 'OUTLIER_THRESHOLD_KWH', 10.0)
        )
        
        # Ghi lưu trữ Model Artifact phân tán xuống HDFS
        logger.info(f"Saving Trained GBT Model Artifact to HDFS: {cfg.MODEL_OUTPUT_PATH}")
        training_results['model'].write().overwrite().save(cfg.MODEL_OUTPUT_PATH)
        
        # Ghi kết quả lưu vết dữ liệu dự đoán (Tùy chọn phục vụ kiểm định trực quan)
        if getattr(cfg, 'SAVE_PREDICTIONS', True):
            save_model_predictions(
                training_results['train_predictions'],
                f"{cfg.PREDICTIONS_OUTPUT_PATH}/train",
                logger
            )
            save_model_predictions(
                training_results['test_predictions'],
                f"{cfg.PREDICTIONS_OUTPUT_PATH}/test",
                logger
            )
        
        # Gộp thông tin cấu hình thực nghiệm phục vụ hàm lưu CSV thống kê
        experiment_config = dict(cfg.TRAIN_CONFIG)
        experiment_config['experiment_name'] = cfg.APP_NAME
        
        # Lưu file tổng hợp CSV
        save_training_metrics_csv(
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            validation_results=validation_results,
            training_time=training_results['training_time'],
            dataset_size=training_results['dataset_size'],
            config=experiment_config,
            metrics_log_path=cfg.TRAINING_METRICS_LOG,
            logger=logger
        )
        
        # Đóng gói toàn vẹn siêu dữ liệu và sai số ghi xuống tệp tin JSON
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'app_name': cfg.APP_NAME,
            'dataset_size': training_results['dataset_size'],
            'train_rows': training_results['train_size'],
            'test_rows': training_results['test_size'],
            'training_time_sec': training_results['training_time'],
            'train_metrics': {k: float(v) if isinstance(v, (int, float)) else v for k, v in train_metrics.items()},
            'test_metrics': {k: float(v) if isinstance(v, (int, float)) else v for k, v in test_metrics.items()},
            'differences': differences,
            'validation_results': {k: float(v) if isinstance(v, (int, float)) else v for k, v in validation_results.items()},
            'config': cfg.TRAIN_CONFIG
        }
        
        # Lưu file chi tiết JSON
        save_training_metadata(cfg, metadata, logger)
        
        # Xuất thống kê báo cáo hoàn thành Job
        logger.info("\n" + "=" * 70)
        logger.info("TRAINING PIPELINE JOB COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)
        logger.info(f"\n FINAL TRAINING SUMMARY:")
        logger.info(f"   Dataset Size  : {training_results['dataset_size']:,} sessions")
        logger.info(f"   Split Balance : Train={training_results['train_size']:,} | Test={training_results['test_size']:,}")
        logger.info(f"   Execution Time: {training_results['training_time']:.2f} seconds")
        logger.info(f"\n GBT TEST SET ERROR METRICS (kWh):")
        logger.info(f"   Mean Absolute Error (MAE) : {test_metrics['mae_kwh']:.4f}")
        logger.info(f"   Root Mean Sq. Error (RMSE): {test_metrics['rmse_kwh']:.4f}")
        logger.info(f"   Coefficient of Det. (R²)  : {test_metrics['r2_kwh']:.4f}")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"Training job failed critically during execution: {str(e)}", exc_info=True)
        sys.exit(1)
    finally:
        if spark is not None:
            spark.stop()
            logger.info("Spark session cleanly disconnected.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: spark-submit src/jobs/acn_training_model.py <config_module>")
        sys.exit(1)
        
    main(sys.argv[1])