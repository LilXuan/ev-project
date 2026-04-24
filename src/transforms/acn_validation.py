from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)


def validate_duration(df: DataFrame) -> Dict[str, int]:
    """Basic data quality checks for duration features"""
    checks = {}
    
    if "duration" in df.columns:
        checks["invalid_duration"] = df.filter(F.col("duration") < 0).count()
    else:
        checks["invalid_duration"] = -1
    
    if "charging_duration" in df.columns:
        checks["invalid_charging_duration"] = df.filter(F.col("charging_duration") < 0).count()
    else:
        checks["invalid_charging_duration"] = -1
    
    return checks


def validate_temporal_features(df: DataFrame) -> Dict[str, int]:
    """Validate temporal feature correctness"""
    checks = {}

    # Range checks
    if "hour" in df.columns:
        checks["invalid_hour"] = df.filter((F.col("hour") < 0) | (F.col("hour") > 23)).count()
    else:
        checks["invalid_hour"] = -1
    
    if "day_of_week" in df.columns:
        checks["invalid_day_of_week"] = df.filter((F.col("day_of_week") < 0) | (F.col("day_of_week") > 6)).count()
    else:
        checks["invalid_day_of_week"] = -1
    
    if "month" in df.columns:
        checks["invalid_month"] = df.filter((F.col("month") < 1) | (F.col("month") > 12)).count()
    else:
        checks["invalid_month"] = -1

    # Weekend consistency
    if "day_of_week" in df.columns and "is_weekend" in df.columns:
        checks["weekend_mismatch"] = df.filter(
            ((F.col("day_of_week") >= 5) & (F.col("is_weekend") != 1)) |
            ((F.col("day_of_week") < 5) & (F.col("is_weekend") != 0))
        ).count()
    else:
        checks["weekend_mismatch"] = -1

    # Holiday consistency
    if "month" in df.columns and "is_holiday" in df.columns:
        checks["holiday_mismatch"] = df.filter(
            ((F.col("month").isin([12, 1])) & (F.col("is_holiday") != 1)) |
            ((~F.col("month").isin([12, 1])) & (F.col("is_holiday") != 0))
        ).count()
    else:
        checks["holiday_mismatch"] = -1

    return checks


def validate_lag_features(df: DataFrame, lags: Tuple[int, ...] = (1, 2, 3)) -> Dict[str, int]:
    """Validate lag feature correctness"""
    checks = {}

    # Null counts (expected: first N rows null)
    for lag_step in lags:
        col_name = f"lag_{lag_step}_log"
        if col_name in df.columns:
            null_count = df.filter(F.col(col_name).isNull()).count()
            checks[f"lag_{lag_step}_nulls"] = null_count
            
            # Negative values check
            invalid = df.filter(F.col(col_name) < 0).count()
            checks[f"lag_{lag_step}_negative"] = invalid
        else:
            checks[f"lag_{lag_step}_nulls"] = -1
            checks[f"lag_{lag_step}_negative"] = -1

    return checks


def validate_rolling_features(df: DataFrame, windows: Tuple[int, ...] = (3, 5)) -> Dict[str, int]:
    """Validate rolling feature correctness"""
    checks = {}

    for w in windows:
        col_name = f"rolling_mean_{w}_log"
        if col_name in df.columns:
            null_count = df.filter(F.col(col_name).isNull()).count()
            checks[f"rolling_{w}_nulls"] = null_count
            
            invalid = df.filter(F.col(col_name) < 0).count()
            checks[f"rolling_{w}_negative"] = invalid
        else:
            checks[f"rolling_{w}_nulls"] = -1
            checks[f"rolling_{w}_negative"] = -1

    return checks


def validate_outliers(df_before: DataFrame, df_after: DataFrame, columns: List[str]) -> Dict[str, Any]:
    """Validate outlier removal impact"""
    checks = {}

    before_count = df_before.count()
    after_count = df_after.count()

    checks["rows_before"] = before_count
    checks["rows_after"] = after_count
    checks["rows_removed"] = before_count - after_count
    checks["pct_removed"] = (
        (before_count - after_count) / before_count if before_count > 0 else 0
    )

    # Negative values check
    for col_name in columns:
        if col_name in df_after.columns:
            neg_count = df_after.filter(F.col(col_name) < 0).count()
            checks[f"{col_name}_negative_after"] = neg_count
        else:
            checks[f"{col_name}_negative_after"] = -1

    return checks


def validate_final_dataset(df: DataFrame, features: List[str], target: str = "kWhDelivered_log") -> Dict[str, int]:
    """Validate final dataset quality"""
    checks = {}

    # Row count
    checks["row_count"] = df.count()

    # Missing values
    for c in features + [target]:
        if c in df.columns:
            nulls = df.filter(F.col(c).isNull()).count()
            checks[f"{c}_nulls"] = nulls
        else:
            checks[f"{c}_nulls"] = -1

    # Negative values check
    if target in df.columns:
        checks["negative_target"] = df.filter(F.col(target) < 0).count()
    else:
        checks["negative_target"] = -1

    return checks


def run_all_validations(df_before: DataFrame, df_after: DataFrame, 
                        config: Dict[str, Any]) -> Dict[str, Any]:
    """Run all validation checks and return results"""
    
    results = {
        "timestamp": F.current_timestamp().cast("string").alias("timestamp"),
        "validations": {}
    }
    
    # Duration validation
    results["validations"]["duration"] = validate_duration(df_after)
    
    # Temporal validation
    results["validations"]["temporal"] = validate_temporal_features(df_after)
    
    # Lag validation
    lags = config.get('lags', (1, 2, 3))
    results["validations"]["lag"] = validate_lag_features(df_after, lags)
    
    # Rolling validation
    windows = config.get('windows', (3, 5))
    results["validations"]["rolling"] = validate_rolling_features(df_after, windows)
    
    # Outlier validation
    outlier_cols = config.get('outlier_cols', ["duration", "charging_duration", "kWhDelivered"])
    results["validations"]["outliers"] = validate_outliers(df_before, df_after, outlier_cols)
    
    # Final dataset validation
    features = config.get('features', [
        'hour', 'day_of_week', 'month', 'season',
        'duration', 'charging_duration', 'charging_duration_log',
        'hour_sin', 'hour_cos', 'day_of_year', 'week_of_year', 'is_holiday',
        'lag_1_log', 'lag_2_log', 'lag_3_log',
        'rolling_mean_3_log', 'rolling_mean_5_log'
    ])
    results["validations"]["final"] = validate_final_dataset(df_after, features)
    
    return results


def log_validation_results(results: Dict[str, Any], logger):
    """Log validation results nicely"""
    
    logger.info("=" * 70)
    logger.info("VALIDATION RESULTS")
    logger.info("=" * 70)
    
    for validation_name, checks in results["validations"].items():
        logger.info(f"\n{validation_name.upper()} VALIDATION:")
        for key, value in checks.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")
    
    # Check for critical issues
    critical_issues = []
    
    # Check outlier removal
    outlier_stats = results["validations"]["outliers"]
    if outlier_stats.get("pct_removed", 0) > 0.1:  # More than 10% removed
        critical_issues.append(f"High outlier removal rate: {outlier_stats['pct_removed']:.2%}")
    
    # Check null counts in final dataset
    final_stats = results["validations"]["final"]
    for key, value in final_stats.items():
        if key.endswith("_nulls") and value > 0:
            critical_issues.append(f"Null values found in {key}: {value} rows")
    
    # Check negative target
    if final_stats.get("negative_target", 0) > 0:
        critical_issues.append(f"Negative target values found: {final_stats['negative_target']} rows")
    
    if critical_issues:
        logger.warning("\n⚠️ CRITICAL ISSUES FOUND:")
        for issue in critical_issues:
            logger.warning(f"  - {issue}")
    else:
        logger.info("\n✅ All validations passed successfully!")
    
    return critical_issues