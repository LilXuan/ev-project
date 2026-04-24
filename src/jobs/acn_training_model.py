#!/usr/bin/env python3
"""
ACN Training Model Job - GBT Regressor for EV Charging Prediction
Simple orchestrator that calls training and validation modules
"""

import sys
import importlib
import json
import csv
import os
from datetime import datetime

from src.utils.spark import create_spark
from src.utils.logger import get_logger
from src.transforms.acn_training import run_training_pipeline
from src.transforms.acn_train_validation import (
    calculate_metrics,
    log_metrics,
    compare_with_paper,
    validate_predictions
)


# def save_training_metadata(output_path: str, metadata: dict, logger):
#     """Save training metadata to JSON file"""
#     try:
#         if output_path.startswith("hdfs://"):
#             local_path = output_path.replace("hdfs://localhost:9000", "")
#             local_path = local_path.rstrip('/') + '/_training_metadata.json'
#         else:
#             local_path = f"{output_path}/_training_metadata.json"
        
#         os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
#         with open(local_path, 'w') as f:
#             json.dump(metadata, f, indent=2, default=str)
        
#         logger.info(f"Training metadata saved to {local_path}")
#     except Exception as e:
#         logger.warning(f"Could not save training metadata: {e}")

def save_training_metadata(output_path: str, metadata: dict, logger):
    """Save training metadata to JSON file"""
    try:
        # Handle HDFS path properly
        if output_path.startswith("hdfs://"):
            # Extract local path from HDFS path
            local_path = output_path.replace("hdfs://localhost:9000", "")
            local_path = local_path.rstrip('/') + '/_training_metadata.json'
            # Remove leading slash for local path
            if local_path.startswith('/'):
                local_path = local_path[1:]
        else:
            local_path = f"{output_path}/_training_metadata.json"
        
        # Create directory with proper permissions
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        with open(local_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Training metadata saved to {local_path}")
    except Exception as e:
        logger.warning(f"Could not save training metadata: {e}")
        # Try to save to current directory as fallback
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
    """Save training metrics to CSV file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    paper_metrics = {'mae_kwh': 2.6697, 'rmse_kwh': 3.9451}
    
    records = []
    
    # Train record
    records.append({
        'timestamp': timestamp,
        'split': 'Train',
        'dataset_size': dataset_size,
        'train_fraction': config.get('train_fraction', 0.8),
        'seed': config.get('seed', 42),
        'training_time_sec': round(training_time, 2),
        'rows': train_metrics.get('rows', 0),
        'mae_log': round(train_metrics.get('mae_log', 0), 4),
        'rmse_log': round(train_metrics.get('rmse_log', 0), 4),
        'r2_log': round(train_metrics.get('r2_log', 0), 4),
        'mae_kwh': round(train_metrics.get('mae_kwh', 0), 4),
        'rmse_kwh': round(train_metrics.get('rmse_kwh', 0), 4),
        'r2_kwh': round(train_metrics.get('r2_kwh', 0), 4),
        'diff_mae_kwh': round(train_metrics.get('mae_kwh', 0) - paper_metrics['mae_kwh'], 4),
        'diff_rmse_kwh': round(train_metrics.get('rmse_kwh', 0) - paper_metrics['rmse_kwh'], 4),
        'mean_pct_error': None,
        'outlier_pct': None
    })
    
    # Test record
    records.append({
        'timestamp': timestamp,
        'split': 'Test',
        'dataset_size': dataset_size,
        'train_fraction': config.get('train_fraction', 0.8),
        'seed': config.get('seed', 42),
        'training_time_sec': round(training_time, 2),
        'rows': test_metrics.get('rows', 0),
        'mae_log': round(test_metrics.get('mae_log', 0), 4),
        'rmse_log': round(test_metrics.get('rmse_log', 0), 4),
        'r2_log': round(test_metrics.get('r2_log', 0), 4),
        'mae_kwh': round(test_metrics.get('mae_kwh', 0), 4),
        'rmse_kwh': round(test_metrics.get('rmse_kwh', 0), 4),
        'r2_kwh': round(test_metrics.get('r2_kwh', 0), 4),
        'diff_mae_kwh': round(test_metrics.get('mae_kwh', 0) - paper_metrics['mae_kwh'], 4),
        'diff_rmse_kwh': round(test_metrics.get('rmse_kwh', 0) - paper_metrics['rmse_kwh'], 4),
        'mean_pct_error': None,
        'outlier_pct': None
    })
    
    # Validation record
    if validation_results:
        records.append({
            'timestamp': timestamp,
            'split': 'Validation',
            'dataset_size': dataset_size,
            'train_fraction': config.get('train_fraction', 0.8),
            'seed': config.get('seed', 42),
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
    
    # Save to CSV
    os.makedirs(os.path.dirname(metrics_log_path), exist_ok=True)
    file_exists = os.path.exists(metrics_log_path)
    
    with open(metrics_log_path, 'a') as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        if not file_exists:
            writer.writeheader()
        for record in records:
            writer.writerow(record)
    
    logger.info(f"✅ Training metrics saved to {metrics_log_path}")


def save_model_predictions(predictions_df, output_path: str, logger, mode: str = "overwrite"):
    """Save model predictions to Parquet"""
    try:
        logger.info(f"Saving predictions to {output_path}")
        
        if hasattr(predictions_df, 'repartition'):
            predictions_df = predictions_df.repartition(4)
        
        predictions_df.write.mode(mode).parquet(output_path)
        logger.info(f"✅ Predictions saved to {output_path}")
        
        count = predictions_df.count()
        logger.info(f"   Saved {count:,} prediction rows")
    except Exception as e:
        logger.error(f"Failed to save predictions: {e}")
        raise


def main(config_module: str):
    """Main training job - simple orchestrator"""
    
    cfg = importlib.import_module(config_module)
    logger = get_logger(f"{cfg.APP_NAME}_Training")
    
    spark = None
    try:
        logger.info("=" * 70)
        logger.info(f"Starting {cfg.APP_NAME} Training Job")
        logger.info("=" * 70)
        
        # Initialize Spark
        spark = create_spark(f"{cfg.APP_NAME}_Training", getattr(cfg, 'SPARK_CONF', {}))
        
        # Load preprocessed data
        logger.info(f"Loading preprocessed data from {cfg.OUTPUT_PATH}")
        df = spark.read.parquet(cfg.OUTPUT_PATH)
        total_rows = df.count()
        logger.info(f"Loaded {total_rows:,} rows, {len(df.columns)} columns")
        
        # Optional subset to match paper's size
        if getattr(cfg, 'USE_PAPER_SUBSET', False):
            target_count = getattr(cfg, 'PAPER_SUBSET_SIZE', 14496)
            fraction = target_count / total_rows
            logger.info(f"⚠️ Subsetting to paper's size: {target_count:,} sessions")
            df = df.sample(fraction=fraction, seed=cfg.TRAIN_CONFIG.get('seed', 42))
            df = df.cache()
            logger.info(f"   After subset: {df.count():,} sessions")
        
        # === CALL TRAINING MODULE ===
        training_results = run_training_pipeline(
            df=df,
            feature_cols=cfg.TRAIN_CONFIG.get('feature_cols', cfg.FEATURES),
            target_col=cfg.TRAIN_CONFIG.get('target_col', 'kWhDelivered_log'),
            train_fraction=cfg.TRAIN_CONFIG.get('train_fraction', 0.8),
            seed=cfg.TRAIN_CONFIG.get('seed', 42),
            max_depth=cfg.TRAIN_CONFIG.get('max_depth', 10),
            max_iter=cfg.TRAIN_CONFIG.get('max_iter', 100),
            use_scaler=True,
            cache_data=True,
            return_predictions=True
        )
        
        # === CALL VALIDATION MODULE ===
        # Calculate metrics
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
        
        # Log metrics
        log_metrics(train_metrics, "TRAIN")
        log_metrics(test_metrics, "TEST")
        
        # Compare with paper
        # Import các hàm validation (nên để ở đầu file)
        from src.transforms.acn_train_validation import print_comparison_table, compare_with_paper_full
        
        # In bảng kết quả giống paper
        print_comparison_table(train_metrics, test_metrics, model_name="GBT")
        
        # So sánh chi tiết với paper's XGBoost và LightGBM
        compare_with_paper_full(test_metrics)
        # ========== KẾT THÚC PHẦN THÊM ==========
        
        # Compare with paper (basic)
        differences = compare_with_paper(test_metrics)
        
        # Validate predictions
        validation_results = validate_predictions(
            training_results['test_predictions'],
            threshold_kwh=getattr(cfg, 'OUTLIER_THRESHOLD_KWH', 10.0)
        )
        
        # Save model
        logger.info(f"Saving model to {cfg.MODEL_OUTPUT_PATH}")
        training_results['model'].write().overwrite().save(cfg.MODEL_OUTPUT_PATH)
        
        # Save predictions (optional)
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
        
        # Save metrics
        save_training_metrics_csv(
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            validation_results=validation_results,
            training_time=training_results['training_time'],
            dataset_size=training_results['dataset_size'],
            config=cfg.TRAIN_CONFIG,
            metrics_log_path=cfg.TRAINING_METRICS_LOG,
            logger=logger
        )
        
        # Save metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'app_name': cfg.APP_NAME,
            'dataset_size': training_results['dataset_size'],
            'train_rows': training_results['train_size'],
            'test_rows': training_results['test_size'],
            'training_time_sec': training_results['training_time'],
            'train_metrics': {k: float(v) if isinstance(v, (int, float)) else v 
                            for k, v in train_metrics.items()},
            'test_metrics': {k: float(v) if isinstance(v, (int, float)) else v 
                           for k, v in test_metrics.items()},
            'differences': differences,
            'validation_results': {k: float(v) if isinstance(v, (int, float)) else v 
                                 for k, v in validation_results.items()},
            'config': cfg.TRAIN_CONFIG
        }
        save_training_metadata(cfg.TRAINING_OUTPUT_PATH, metadata, logger)
        
        # Final summary
        logger.info("\n" + "=" * 70)
        logger.info("TRAINING JOB COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)
        logger.info(f"\n📊 FINAL SUMMARY:")
        logger.info(f"  Dataset: {training_results['dataset_size']:,} sessions")
        logger.info(f"  Train: {training_results['train_size']:,} | Test: {training_results['test_size']:,}")
        logger.info(f"  Training time: {training_results['training_time']:.2f}s")
        logger.info(f"\n📊 TEST SET PERFORMANCE (kWh):")
        logger.info(f"  MAE: {test_metrics['mae_kwh']:.4f}")
        logger.info(f"  RMSE: {test_metrics['rmse_kwh']:.4f}")
        logger.info(f"  R²: {test_metrics['r2_kwh']:.4f}")
        
    except Exception as e:
        logger.error(f"Training job failed: {str(e)}", exc_info=True)
        sys.exit(1)
    finally:
        if spark is not None:
            spark.stop()
            logger.info("Spark session stopped")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python acn_training_model.py <config_module>")
        print("Example: python acn_training_model.py configs.config_caltech")
        sys.exit(1)
    
    main(sys.argv[1])
    
