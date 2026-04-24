"""
ACN Training Pipeline - Complete Orchestrator
Combines training and validation logic
"""

import logging
from typing import Dict, Any, List, Optional

from pyspark.sql import DataFrame
from src.transforms.acn_training import run_training_pipeline
from src.transforms.acn_train_validation import (
    evaluate_on_both_sets,
    compare_with_paper,
    validate_predictions,
    calculate_metrics,
    log_metrics
)

logger = logging.getLogger(__name__)


def run_complete_pipeline(
    df: DataFrame,
    feature_cols: List[str],
    target_col: str = "kWhDelivered_log",
    train_fraction: float = 0.8,
    seed: int = 42,
    max_depth: int = 10,
    max_iter: int = 100,
    use_scaler: bool = True,
    cache_data: bool = True,
    validate_outliers: bool = True,
    outlier_threshold: float = 10.0
) -> Dict[str, Any]:
    """
    Complete training and validation pipeline
    
    Args:
        df: Preprocessed DataFrame
        feature_cols: List of feature column names
        target_col: Target column name
        train_fraction: Training fraction
        seed: Random seed
        max_depth: GBT max depth
        max_iter: GBT max iterations
        use_scaler: Whether to use StandardScaler
        cache_data: Whether to cache DataFrames
        validate_outliers: Whether to perform outlier validation
        outlier_threshold: Threshold for outlier detection (kWh)
    
    Returns:
        Dictionary with all results
    """
    # Step 1: Train model and get predictions
    training_results = run_training_pipeline(
        df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        train_fraction=train_fraction,
        seed=seed,
        max_depth=max_depth,
        max_iter=max_iter,
        use_scaler=use_scaler,
        cache_data=cache_data,
        return_predictions=True
    )
    
    # Step 2: Calculate metrics on both sets
    train_metrics = calculate_metrics(
        training_results['train_predictions'],
        target_col,
        "prediction"
    )
    test_metrics = calculate_metrics(
        training_results['test_predictions'],
        target_col,
        "prediction"
    )
    
    # Log metrics
    log_metrics(train_metrics, "TRAIN")
    log_metrics(test_metrics, "TEST")
    
    # Step 3: Compare with paper
    differences = compare_with_paper(test_metrics)
    
    # Step 4: Validate predictions (optional)
    validation_results = None
    if validate_outliers:
        validation_results = validate_predictions(
            training_results['test_predictions'],
            threshold_kwh=outlier_threshold
        )
    
    # Step 5: Prepare final results
    results = {
        'model': training_results['model'],
        'train_df': training_results['train_df'],
        'test_df': training_results['test_df'],
        'train_predictions': training_results['train_predictions'],
        'test_predictions': training_results['test_predictions'],
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'differences': differences,
        'training_time': training_results['training_time'],
        'dataset_size': training_results['dataset_size'],
        'train_size': training_results['train_size'],
        'test_size': training_results['test_size'],
        'validation_results': validation_results
    }
    
    
    # Analyze target distribution
    logger.info("\n🔍 Analyzing target variable distribution...")
    target_stats = analyze_target_distribution(df, target_col.replace('_log', ''))
    
    # Check if log transformation is appropriate
    if target_stats['original_skewness'] > 1:
        logger.info("✅ Using log transformation for skewed target variable")
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("COMPLETE PIPELINE FINISHED")
    logger.info("=" * 70)
    logger.info(f"\n📊 FINAL TEST METRICS (kWh):")
    logger.info(f"  MAE: {test_metrics['mae_kwh']:.4f} | RMSE: {test_metrics['rmse_kwh']:.4f} | R²: {test_metrics['r2_kwh']:.4f}")
    
    return results