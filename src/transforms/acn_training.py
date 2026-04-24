"""
ACN Training Module - GBT Regressor
Handles training pipeline, model building, and predictions
"""

import time
import logging
from typing import Dict, Any, Tuple, List, Optional

from pyspark.sql import DataFrame
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, exp

logger = logging.getLogger(__name__)


def train_test_split_random(
    df: DataFrame, 
    train_fraction: float = 0.8, 
    seed: int = 42,
    drop_time_col: bool = True
) -> Tuple[DataFrame, DataFrame]:
    """
    Random split (Method 1 from notebook) - matches paper's results
    
    Args:
        df: Final preprocessed DataFrame with connectionTime_utc
        train_fraction: Training fraction (default 0.8)
        seed: Random seed for reproducibility
        drop_time_col: Whether to drop connectionTime_utc
    
    Returns:
        train_df, test_df
    """
    logger.info(f"Train/test split: {train_fraction:.1%} train, {1-train_fraction:.1%} test (seed={seed})")
    
    total_rows = df.count()
    train_size = int(total_rows * train_fraction)
    test_size = total_rows - train_size
    
    logger.info(f"Total: {total_rows:,} | Train: {train_size:,} | Test: {test_size:,}")
    
    # Random split with fixed seed
    train_df = df.sample(fraction=train_fraction, seed=seed)
    test_df = df.subtract(train_df)
    
    # Drop connectionTime_utc (not used for training)
    if drop_time_col and "connectionTime_utc" in train_df.columns:
        train_df = train_df.drop("connectionTime_utc")
        test_df = test_df.drop("connectionTime_utc")
    
    return train_df, test_df


def build_gbt_pipeline(
    feature_cols: List[str],
    target_col: str = "kWhDelivered_log",
    max_depth: int = 10,
    max_iter: int = 100,
    seed: int = 42,
    use_scaler: bool = True
) -> Pipeline:
    """
    Build GBT pipeline with VectorAssembler, optional StandardScaler, and GBTRegressor
    
    Args:
        feature_cols: List of feature column names
        target_col: Target column name
        max_depth: Maximum tree depth
        max_iter: Number of boosting iterations
        seed: Random seed
        use_scaler: Whether to use StandardScaler
    
    Returns:
        Pipeline object
    """
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    
    stages = [assembler]
    
    if use_scaler:
        scaler = StandardScaler(
            inputCol="features", 
            outputCol="scaled_features", 
            withStd=True, 
            withMean=True
        )
        features_col_for_gbt = "scaled_features"
        stages.append(scaler)
    else:
        features_col_for_gbt = "features"
    
    gbt = GBTRegressor(
        featuresCol=features_col_for_gbt,
        labelCol=target_col,
        maxDepth=max_depth,
        maxIter=max_iter,
        seed=seed
    )
    
    stages.append(gbt)
    
    return Pipeline(stages=stages)


def train_model(
    train_df: DataFrame,
    pipeline: Pipeline,
    cache_train: bool = True
) -> Tuple[Pipeline, float]:
    """
    Train GBT model and measure training time
    
    Args:
        train_df: Training DataFrame
        pipeline: Pipeline object
        cache_train: Whether to cache training data
    
    Returns:
        trained_model, training_time_seconds
    """
    logger.info("=" * 50)
    logger.info("Training GBT Regressor")
    logger.info("=" * 50)
    
    # Cache training data if needed
    if cache_train:
        train_df = train_df.cache()
        train_df.count()  # Force cache
    
    start_time = time.time()
    model = pipeline.fit(train_df)
    training_time = time.time() - start_time
    
    logger.info(f"✅ Training completed in {training_time:.2f} seconds")
    logger.info(f"   Rows: {train_df.count():,}")
    
    return model, training_time


def add_original_scale_predictions(
    predictions_df: DataFrame,
    target_col: str = "kWhDelivered_log"
) -> DataFrame:
    """
    Add original scale predictions (kWh) to predictions DataFrame
    
    Args:
        predictions_df: DataFrame with actual and predicted values on log scale
        target_col: Target column name
    
    Returns:
        DataFrame with additional columns: actual_kwh, pred_kwh
    """
    return predictions_df \
        .withColumn("actual_kwh", exp(col(target_col)) - 1) \
        .withColumn("pred_kwh", exp(col("prediction")) - 1)


def run_training_pipeline(
    df: DataFrame,
    feature_cols: List[str],
    target_col: str = "kWhDelivered_log",
    train_fraction: float = 0.8,
    seed: int = 42,
    max_depth: int = 10,
    max_iter: int = 100,
    use_scaler: bool = True,
    cache_data: bool = True,
    return_predictions: bool = False
) -> Dict[str, Any]:
    """
    Main training pipeline orchestrator
    
    Args:
        df: Preprocessed DataFrame (with connectionTime_utc)
        feature_cols: List of feature column names
        target_col: Target column name
        train_fraction: Training fraction
        seed: Random seed
        max_depth: GBT max depth
        max_iter: GBT max iterations
        use_scaler: Whether to use StandardScaler
        cache_data: Whether to cache DataFrames
        return_predictions: Whether to return predictions DataFrame
    
    Returns:
        Dictionary with all results (including train and test metrics)
    """
    logger.info("=" * 70)
    logger.info("ACN TRAINING PIPELINE - GBT Regressor")
    logger.info("=" * 70)
    logger.info(f"Dataset size: {df.count():,} rows")
    logger.info(f"Features: {len(feature_cols)}")
    logger.info(f"Target: {target_col}")
    
    # Step 1: Train/test split
    train_df, test_df = train_test_split_random(df, train_fraction, seed)
    
    if cache_data:
        train_df = train_df.cache()
        test_df = test_df.cache()
        train_df.count()  # Force cache
        test_df.count()
    
    # Step 2: Build pipeline
    pipeline = build_gbt_pipeline(
        feature_cols=feature_cols,
        target_col=target_col,
        max_depth=max_depth,
        max_iter=max_iter,
        seed=seed,
        use_scaler=use_scaler
    )
    
    # Step 3: Train model
    model, training_time = train_model(train_df, pipeline, cache_train=cache_data)
    
    # Step 4: Get predictions
    train_predictions = model.transform(train_df)
    test_predictions = model.transform(test_df)
    
    # Step 5: Add original scale predictions
    train_predictions_orig = add_original_scale_predictions(train_predictions, target_col)
    test_predictions_orig = add_original_scale_predictions(test_predictions, target_col)
    
    # Step 6: Prepare results
    results = {
        'model': model,
        'train_df': train_df,
        'test_df': test_df,
        'train_predictions': train_predictions_orig,
        'test_predictions': test_predictions_orig,
        'training_time': training_time,
        'dataset_size': df.count(),
        'train_size': train_df.count(),
        'test_size': test_df.count(),
        'target_col': target_col
    }
    
    if not return_predictions:
        # Remove predictions from results if not requested (saves memory)
        results.pop('train_predictions', None)
        results.pop('test_predictions', None)
    
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING PIPELINE COMPLETED")
    logger.info("=" * 70)
    logger.info(f"  Dataset: {results['dataset_size']:,} sessions")
    logger.info(f"  Train: {results['train_size']:,} | Test: {results['test_size']:,}")
    logger.info(f"  Training time: {training_time:.2f}s")
    
    return results
