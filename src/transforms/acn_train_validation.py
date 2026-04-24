"""
ACN Validation Module - Metrics Calculation and Comparison
Handles evaluation metrics, paper comparison, and validation logic
"""

import logging
from typing import Dict, Any, Tuple, Optional

from pyspark.sql import DataFrame
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, avg, sqrt, pow, abs, sum as spark_sum

logger = logging.getLogger(__name__)


def calculate_metrics(
    predictions_df: DataFrame,
    target_col: str = "kWhDelivered_log",
    prediction_col: str = "prediction"
) -> Dict[str, float]:
    """
    Calculate comprehensive metrics on both log and original scales
    """
    # Add original scale predictions if not present
    if "actual_kwh" not in predictions_df.columns or "pred_kwh" not in predictions_df.columns:
        from pyspark.sql.functions import exp
        predictions_df = predictions_df \
            .withColumn("actual_kwh", exp(col(target_col)) - 1) \
            .withColumn("pred_kwh", exp(col(prediction_col)) - 1)
    
    # Calculate error metrics
    error_df = predictions_df \
        .withColumn("error", col("actual_kwh") - col("pred_kwh")) \
        .withColumn("abs_error", abs(col("error"))) \
        .withColumn("squared_error", pow(col("error"), 2))
    
    # Calculate metrics on original scale (kWh)
    metrics_orig = error_df.select(
        avg("abs_error").alias("mae_kwh"),
        avg("squared_error").alias("mse_kwh"),
        sqrt(avg("squared_error")).alias("rmse_kwh")
    ).collect()[0]
    
    # VALIDATION CHECK: Nếu MAE quá nhỏ (< 1.0 kWh), cảnh báo
    if metrics_orig['mae_kwh'] < 1.0:
        logger.warning(f"⚠️ MAE = {metrics_orig['mae_kwh']:.4f} kWh is unusually small!")
        logger.warning("   Possible issue: Target might be in Wh instead of kWh")
        logger.warning("   Expected MAE range: 2.5 - 3.5 kWh for EV charging data")
    
    # Calculate R2 on original scale
    actual_mean = predictions_df.select(avg("actual_kwh")).collect()[0][0]
    ss_res = error_df.select(spark_sum("squared_error").alias("ss_res")).collect()[0][0]
    ss_tot = predictions_df.select(
        spark_sum(pow(col("actual_kwh") - actual_mean, 2)).alias("ss_tot")
    ).collect()[0][0]
    r2_kwh = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    
    # Calculate log scale metrics
    evaluator_rmse = RegressionEvaluator(labelCol=target_col, predictionCol=prediction_col, metricName="rmse")
    evaluator_mse = RegressionEvaluator(labelCol=target_col, predictionCol=prediction_col, metricName="mse")
    evaluator_mae = RegressionEvaluator(labelCol=target_col, predictionCol=prediction_col, metricName="mae")
    evaluator_r2 = RegressionEvaluator(labelCol=target_col, predictionCol=prediction_col, metricName="r2")
    
    metrics = {
        # Log scale metrics
        'rmse_log': evaluator_rmse.evaluate(predictions_df),
        'mse_log': evaluator_mse.evaluate(predictions_df),
        'mae_log': evaluator_mae.evaluate(predictions_df),
        'r2_log': evaluator_r2.evaluate(predictions_df),
        # Original scale metrics (kWh)
        'mae_kwh': metrics_orig['mae_kwh'],
        'mse_kwh': metrics_orig['mse_kwh'],  # ✅ Đã có MSE
        'rmse_kwh': metrics_orig['rmse_kwh'],
        'r2_kwh': r2_kwh,
        # Additional
        'rows': predictions_df.count()
    }
    
    return metrics


def log_metrics(metrics: Dict[str, float], dataset_name: str = "TEST") -> None:
    """
    Log metrics in a formatted way
    
    Args:
        metrics: Dictionary of metrics
        dataset_name: Name of the dataset (TRAIN/TEST)
    """
    logger.info(f"\n📊 {dataset_name} SET METRICS ON LOG SCALE:")
    logger.info(f"  MAE: {metrics['mae_log']:.4f} | MSE: {metrics['mse_log']:.4f} | RMSE: {metrics['rmse_log']:.4f} | R²: {metrics['r2_log']:.4f}")
    
    logger.info(f"\n📊 {dataset_name} SET METRICS ON ORIGINAL SCALE (kWh):")
    logger.info(f"  MAE: {metrics['mae_kwh']:.4f} kWh")
    logger.info(f"  MSE: {metrics['mse_kwh']:.4f} kWh²")  # ✅ Thêm dòng MSE
    logger.info(f"  RMSE: {metrics['rmse_kwh']:.4f} kWh")
    logger.info(f"  R²: {metrics['r2_kwh']:.4f}")
    logger.info(f"  Rows: {metrics['rows']:,}")
    
    


def evaluate_model(
    model,  # PipelineModel
    test_df: DataFrame,
    target_col: str = "kWhDelivered_log"
) -> Tuple[Dict[str, float], float, DataFrame]:
    """
    Evaluate model on test set with multiple metrics on both log and original scales
    
    Args:
        model: Trained pipeline model
        test_df: Test DataFrame
        target_col: Target column name
    
    Returns:
        metrics_dict, inference_time_seconds, predictions_with_actual
    """
    logger.info("=" * 50)
    logger.info("Evaluating Model")
    logger.info("=" * 50)
    
    # Predict
    import time
    start_time = time.time()
    predictions = model.transform(test_df)
    inference_time = time.time() - start_time
    
    # Add original scale predictions
    from pyspark.sql.functions import exp
    predictions_orig = predictions \
        .withColumn("actual_kwh", exp(col(target_col)) - 1) \
        .withColumn("pred_kwh", exp(col("prediction")) - 1)
    
    # Calculate metrics
    metrics = calculate_metrics(predictions_orig, target_col, "prediction")
    metrics['inference_time_sec'] = inference_time
    
    # Log metrics
    log_metrics(metrics, "TEST")
    
    return metrics, inference_time, predictions_orig


def evaluate_on_both_sets(
    model,  # PipelineModel
    train_df: DataFrame,
    test_df: DataFrame,
    target_col: str = "kWhDelivered_log"
) -> Tuple[Dict[str, float], Dict[str, float], float, float]:
    """
    Evaluate model on both train and test sets
    
    Args:
        model: Trained pipeline model
        train_df: Training DataFrame
        test_df: Test DataFrame
        target_col: Target column name
    
    Returns:
        train_metrics_dict, test_metrics_dict, train_inference_time, test_inference_time
    """
    logger.info("=" * 50)
    logger.info("Evaluating Model on Both Train and Test Sets")
    logger.info("=" * 50)
    
    # Predict on train set
    logger.info("Predicting on TRAIN set...")
    import time
    start_time = time.time()
    train_predictions = model.transform(train_df)
    train_inference_time = time.time() - start_time
    
    # Predict on test set
    logger.info("Predicting on TEST set...")
    start_time = time.time()
    test_predictions = model.transform(test_df)
    test_inference_time = time.time() - start_time
    
    # Add original scale predictions
    from pyspark.sql.functions import exp
    train_predictions_orig = train_predictions \
        .withColumn("actual_kwh", exp(col(target_col)) - 1) \
        .withColumn("pred_kwh", exp(col("prediction")) - 1)
    
    test_predictions_orig = test_predictions \
        .withColumn("actual_kwh", exp(col(target_col)) - 1) \
        .withColumn("pred_kwh", exp(col("prediction")) - 1)
    
    # Calculate metrics
    train_metrics = calculate_metrics(train_predictions_orig, target_col, "prediction")
    test_metrics = calculate_metrics(test_predictions_orig, target_col, "prediction")
    
    # Log metrics
    log_metrics(train_metrics, "TRAIN")
    log_metrics(test_metrics, "TEST")
    
    return train_metrics, test_metrics, train_inference_time, test_inference_time


def compare_with_paper(
    metrics: Dict[str, float], 
    paper_metrics: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Compare results with paper's reported metrics
    
    Args:
        metrics: Current model metrics
        paper_metrics: Paper's metrics (default: XGBoost from paper)
    
    Returns:
        Dictionary with differences
    """
    if paper_metrics is None:
        paper_metrics = {
            'mae_kwh': 2.6697,
            'rmse_kwh': 3.9451
        }
    
    differences = {
        'diff_mae_kwh': metrics['mae_kwh'] - paper_metrics['mae_kwh'],
        'diff_rmse_kwh': metrics['rmse_kwh'] - paper_metrics['rmse_kwh'],
        'pct_diff_mae': ((metrics['mae_kwh'] - paper_metrics['mae_kwh']) / paper_metrics['mae_kwh']) * 100,
        'pct_diff_rmse': ((metrics['rmse_kwh'] - paper_metrics['rmse_kwh']) / paper_metrics['rmse_kwh']) * 100
    }
    
    logger.info("\n" + "=" * 50)
    logger.info("COMPARISON WITH PAPER'S XGBoost")
    logger.info("=" * 50)
    logger.info(f"  Metric          Paper XGBoost      Our GBT            Difference")
    logger.info(f"  {'-' * 65}")
    logger.info(f"  MAE (kWh)        {paper_metrics['mae_kwh']:.4f}              {metrics['mae_kwh']:.4f}               {differences['diff_mae_kwh']:+.4f}")
    logger.info(f"  RMSE (kWh)       {paper_metrics['rmse_kwh']:.4f}              {metrics['rmse_kwh']:.4f}               {differences['diff_rmse_kwh']:+.4f}")
    
    return differences


def validate_predictions(
    predictions_df: DataFrame,
    threshold_kwh: float = 10.0
) -> Dict[str, Any]:
    """
    Validate predictions by checking error distributions and outliers
    
    Args:
        predictions_df: DataFrame with actual_kwh and pred_kwh columns
        threshold_kwh: Threshold for outlier detection (default 10 kWh)
    
    Returns:
        Dictionary with validation statistics
    """
    logger.info("=" * 50)
    logger.info("Validating Predictions")
    logger.info("=" * 50)
    
    # Calculate absolute error
    predictions_with_error = predictions_df \
        .withColumn("abs_error", abs(col("actual_kwh") - col("pred_kwh"))) \
        .withColumn("pct_error", (col("abs_error") / (col("actual_kwh") + 1e-6)) * 100)
    
    # Calculate statistics
    stats = predictions_with_error.select(
        avg("abs_error").alias("mean_abs_error"),
        sqrt(avg(pow(col("abs_error"), 2))).alias("rmse_error"),
        avg("pct_error").alias("mean_pct_error"),
        spark_sum((col("abs_error") > threshold_kwh).cast("int")).alias("outliers_count")
    ).collect()[0]
    
    total_rows = predictions_with_error.count()
    outlier_pct = (stats['outliers_count'] / total_rows) * 100 if total_rows > 0 else 0
    
    validation_results = {
        'mean_absolute_error': stats['mean_abs_error'],
        'rmse_error': stats['rmse_error'],
        'mean_percentage_error': stats['mean_pct_error'],
        'outliers_count': stats['outliers_count'],
        'outlier_percentage': outlier_pct,
        'total_rows': total_rows,
        'threshold_kwh': threshold_kwh
    }
    
    logger.info(f"  Mean Absolute Error: {validation_results['mean_absolute_error']:.4f} kWh")
    logger.info(f"  RMSE: {validation_results['rmse_error']:.4f} kWh")
    logger.info(f"  Mean Percentage Error: {validation_results['mean_percentage_error']:.2f}%")
    logger.info(f"  Outliers (> {threshold_kwh} kWh): {validation_results['outliers_count']:,} ({validation_results['outlier_percentage']:.2f}%)")
    
    return validation_results


def print_comparison_table(
    train_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    model_name: str = "GBT"
) -> None:
    """
    Print results in a table format similar to paper
    
    Args:
        train_metrics: Training metrics
        test_metrics: Test metrics
        model_name: Name of the model
    """
    logger.info("\n" + "=" * 80)
    logger.info(f"{model_name} MODEL PERFORMANCE")
    logger.info("=" * 80)
    logger.info(f"{'Split':<10} {'MAE (kWh)':<15} {'MSE (kWh²)':<15} {'RMSE (kWh)':<15} {'R²':<10}")
    logger.info("-" * 80)
    logger.info(f"{'Train':<10} {train_metrics['mae_kwh']:<15.4f} {train_metrics['mse_kwh']:<15.4f} {train_metrics['rmse_kwh']:<15.4f} {train_metrics['r2_kwh']:<10.4f}")
    logger.info(f"{'Test':<10} {test_metrics['mae_kwh']:<15.4f} {test_metrics['mse_kwh']:<15.4f} {test_metrics['rmse_kwh']:<15.4f} {test_metrics['r2_kwh']:<10.4f}")
    logger.info("=" * 80)


def compare_with_paper_full(
    metrics: Dict[str, float],
    paper_xgboost: Optional[Dict[str, float]] = None,
    paper_lightgbm: Optional[Dict[str, float]] = None
) -> None:
    """
    Compare results with paper's reported metrics (XGBoost and LightGBM)
    
    Args:
        metrics: Current model metrics (test set)
        paper_xgboost: Paper's XGBoost metrics
        paper_lightgbm: Paper's LightGBM metrics
    """
    if paper_xgboost is None:
        paper_xgboost = {
            'mae_kwh': 2.6697,
            'mse_kwh': 15.5640,  # 3.9451^2
            'rmse_kwh': 3.9451,
            'r2': 0.6463
        }
    
    if paper_lightgbm is None:
        paper_lightgbm = {
            'mae_kwh': 2.6523,
            'mse_kwh': 15.1386,  # 3.8908^2
            'rmse_kwh': 3.8908,
            'r2': 0.6559
        }
    
    logger.info("\n" + "=" * 90)
    logger.info("COMPARISON WITH PAPER'S RESULTS (Test Set)")
    logger.info("=" * 90)
    logger.info(f"{'Metric':<15} {'Paper XGBoost':<18} {'Paper LightGBM':<18} {'Our GBT':<15} {'Diff vs XGB':<12}")
    logger.info("-" * 90)
    logger.info(f"{'MAE (kWh)':<15} {paper_xgboost['mae_kwh']:<18.4f} {paper_lightgbm['mae_kwh']:<18.4f} {metrics['mae_kwh']:<15.4f} {metrics['mae_kwh'] - paper_xgboost['mae_kwh']:+12.4f}")
    logger.info(f"{'MSE (kWh²)':<15} {paper_xgboost['mse_kwh']:<18.4f} {paper_lightgbm['mse_kwh']:<18.4f} {metrics['mse_kwh']:<15.4f} {metrics['mse_kwh'] - paper_xgboost['mse_kwh']:+12.4f}")
    logger.info(f"{'RMSE (kWh)':<15} {paper_xgboost['rmse_kwh']:<18.4f} {paper_lightgbm['rmse_kwh']:<18.4f} {metrics['rmse_kwh']:<15.4f} {metrics['rmse_kwh'] - paper_xgboost['rmse_kwh']:+12.4f}")
    logger.info(f"{'R²':<15} {paper_xgboost['r2']:<18.4f} {paper_lightgbm['r2']:<18.4f} {metrics['r2_kwh']:<15.4f} {metrics['r2_kwh'] - paper_xgboost['r2']:+12.4f}")
    logger.info("=" * 90)
    
    # Đánh giá
    logger.info("\n📊 PERFORMANCE ASSESSMENT:")
    if metrics['mae_kwh'] <= paper_xgboost['mae_kwh'] * 1.05:
        logger.info("  ✅ MAE is within 5% of paper's XGBoost (good)")
    elif metrics['mae_kwh'] <= paper_xgboost['mae_kwh'] * 1.10:
        logger.info("  ⚠️ MAE is within 10% of paper's XGBoost (acceptable)")
    else:
        logger.info("  ❌ MAE deviation > 10% from paper's XGBoost (needs improvement)")
    
    if metrics['r2_kwh'] >= 0.6:
        logger.info("  ✅ R² >= 0.6 (good explanatory power)")
    elif metrics['r2_kwh'] >= 0.5:
        logger.info("  ⚠️ R² between 0.5 and 0.6 (moderate)")
    else:
        logger.info("  ❌ R² < 0.5 (low explanatory power)")
        

def analyze_target_distribution(df: DataFrame, target_col: str = "kWhDelivered"):
    """
    Analyze target variable distribution for model validation
    """
    logger.info("=" * 50)
    logger.info("Target Variable Distribution Analysis")
    logger.info("=" * 50)
    
    # Basic statistics
    stats = df.select(
        min(col(target_col)).alias("min"),
        expr("percentile_approx({}, 0.25)".format(target_col)).alias("q1"),
        expr("percentile_approx({}, 0.5)".format(target_col)).alias("median"),
        expr("percentile_approx({}, 0.75)".format(target_col)).alias("q3"),
        max(col(target_col)).alias("max"),
        avg(col(target_col)).alias("mean"),
        stddev(col(target_col)).alias("stddev")
    ).collect()[0]
    
    # Calculate skewness
    skewness_val = df.select(skewness(target_col)).collect()[0][0]
    
    logger.info(f"\n📊 {target_col} Statistics:")
    logger.info(f"  Min: {stats['min']:.2f} kWh")
    logger.info(f"  Q1: {stats['q1']:.2f} kWh")
    logger.info(f"  Median: {stats['median']:.2f} kWh")
    logger.info(f"  Q3: {stats['q3']:.2f} kWh")
    logger.info(f"  Max: {stats['max']:.2f} kWh")
    logger.info(f"  Mean: {stats['mean']:.2f} kWh")
    logger.info(f"  StdDev: {stats['stddev']:.2f} kWh")
    logger.info(f"  Skewness: {skewness_val:.4f}")
    
    # Interpretation
    if abs(skewness_val) > 1:
        logger.info(f"  ⚠️ Highly skewed distribution (skewness = {skewness_val:.4f})")
        logger.info(f"  ✅ Log transformation is appropriate")
    
    # Check log transformation effect
    df_log = df.withColumn(f"{target_col}_log", log(col(target_col) + 1))
    skewness_log = df_log.select(skewness(f"{target_col}_log")).collect()[0][0]
    logger.info(f"\n📊 After Log Transformation:")
    logger.info(f"  Skewness: {skewness_log:.4f}")
    logger.info(f"  Improvement: {abs(skewness_val) - abs(skewness_log):.4f}")
    
    return {
        'original': dict(stats.asDict()),
        'original_skewness': skewness_val,
        'log_skewness': skewness_log
    }


def validate_predictions_by_range(
    predictions_df: DataFrame,
    actual_col: str = "actual_kwh",
    pred_col: str = "pred_kwh"
):
    """
    Validate predictions across different energy ranges
    """
    logger.info("=" * 50)
    logger.info("Prediction Performance by Energy Range")
    logger.info("=" * 50)
    
    # Create range buckets
    predictions_with_range = predictions_df \
        .withColumn("energy_range", 
            when(col(actual_col) <= 5, "0-5 kWh")
            .when(col(actual_col) <= 10, "5-10 kWh")
            .when(col(actual_col) <= 20, "10-20 kWh")
            .otherwise(">20 kWh")
        )
    
    # Calculate metrics by range
    range_metrics = predictions_with_range.groupBy("energy_range").agg(
        avg(abs(col(actual_col) - col(pred_col))).alias("mae_kwh"),
        sqrt(avg(pow(col(actual_col) - col(pred_col), 2))).alias("rmse_kwh"),
        (avg(abs(col(actual_col) - col(pred_col)) / (col(actual_col) + 1e-6)) * 100).alias("mape_pct"),
        count("*").alias("count")
    ).orderBy("energy_range")
    
    range_metrics.show(truncate=False)
    
    # Check for systematic bias
    logger.info("\n📊 Systematic Bias Check:")
    bias_by_range = predictions_with_range.groupBy("energy_range").agg(
        avg(col(pred_col) - col(actual_col)).alias("mean_bias_kwh"),
        (avg(col(pred_col) - col(actual_col)) / avg(col(actual_col)) * 100).alias("bias_pct")
    ).orderBy("energy_range")
    
    bias_by_range.show(truncate=False)
    
    return range_metrics.toPandas().to_dict('records')