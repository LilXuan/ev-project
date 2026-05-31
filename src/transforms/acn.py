# src/transforms/acn.py
from pyspark.sql import Window
from pyspark.sql import functions as F
from typing import List, Tuple, Dict, Any, Optional

# --- CONFIGURATIONS ---
ID_COLUMNS = [
    '_batch_id', '_id', '_ingest_time',
    'sessionID', 'stationID', 'spaceID',
    'siteID', 'clusterID', 'userID'
]

TIME_COLUMNS = ['connectionTime', 'disconnectTime', 'doneChargingTime']

# --- CORE FUNCTIONS ---

def drop_id_columns(df):
    """Step 1: Remove ID columns"""
    existing = [c for c in ID_COLUMNS if c in df.columns]
    if existing:
        df = df.drop(*existing)
    return df, existing

def convert_time(df):
    """Step 2: Convert to timezone-naive UTC"""
    for c in TIME_COLUMNS:
        if c in df.columns:
            df = df.withColumn(
                f"{c}_utc",
                F.from_utc_timestamp(F.to_timestamp(F.col(c), "yyyy-MM-dd HH:mm:ssXXX"), "UTC")
            ).drop(c)

    for c in ['timezone', 'userInputs']:
        if c in df.columns:
            df = df.drop(c)
    return df

def add_duration_features(df):
    """Step 3: Duration and initial charging_duration_log"""
    # Tính duration (giờ)
    df = df.withColumn("duration", 
        (F.unix_timestamp(F.col("disconnectTime_utc")) - F.unix_timestamp(F.col("connectionTime_utc"))) / 3600
    )
    
    # Tính charging_duration (giờ)
    df = df.withColumn("charging_duration",
        F.when(F.col("doneChargingTime_utc").isNotNull(),
               (F.unix_timestamp(F.col("doneChargingTime_utc")) - F.unix_timestamp(F.col("connectionTime_utc"))) / 3600
        ).otherwise(None)
    )

    # Log transform charging_duration
    df = df.withColumn("charging_duration_log",
        F.when(F.col("charging_duration") > 0, F.log1p(F.col("charging_duration"))).otherwise(None)
    )
    return df

def add_temporal_features(df):
    """Step 4: Temporal features and Cyclical encoding"""
    df = df.withColumn("hour", F.hour(F.col("connectionTime_utc"))) \
           .withColumn("day_of_week", F.dayofweek(F.col("connectionTime_utc")) - 1) \
           .withColumn("month", F.month(F.col("connectionTime_utc"))) \
           .withColumn("year", F.year(F.col("connectionTime_utc"))) \
           .withColumn("day_of_year", F.dayofyear(F.col("connectionTime_utc"))) \
           .withColumn("week_of_year", F.weekofyear(F.col("connectionTime_utc")))

    df = df.withColumn("season",
        F.when(F.col("month").isin([12, 1, 2]), 0)
         .when(F.col("month").isin([3, 4, 5]), 1)
         .when(F.col("month").isin([6, 7, 8]), 2).otherwise(3))

    df = df.withColumn("is_weekend", F.when(F.col("day_of_week") >= 5, 1).otherwise(0))
    df = df.withColumn("is_holiday", F.when(F.col("month").isin([12, 1]), 1).otherwise(0))

    df = df.withColumn("hour_sin", F.sin(2 * F.pi() * F.col("hour") / 24)) \
           .withColumn("hour_cos", F.cos(2 * F.pi() * F.col("hour") / 24))
    return df

def add_lag_features(df, lags=(1, 2, 3)):
    """Step 5: Global lag features"""
    window_spec = Window.orderBy("connectionTime_utc")
    for lag_step in lags:
        df = df.withColumn(f"lag_{lag_step}_log", F.lag("charging_duration_log", lag_step).over(window_spec))
    return df

def add_rolling_features(df, windows=(3, 5)):
    """Step 6: Rolling mean features"""
    base_window = Window.orderBy("connectionTime_utc")
    for w in windows:
        window_spec = base_window.rowsBetween(-w, -1)
        df = df.withColumn(f"rolling_mean_{w}_log", F.avg("charging_duration_log").over(window_spec))
    return df

def compute_iqr_bounds(df, column: str) -> Tuple[float, float]:
    """Helper to compute bounds on CURRENT state of dataframe"""
    quantiles = df.select(column).filter(F.col(column).isNotNull())\
                  .approxQuantile(column, [0.25, 0.75], 0.01)
    if len(quantiles) < 2: return -float('inf'), float('inf')
    q1, q3 = quantiles[0], quantiles[1]
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr

def remove_outliers_iqr(df, columns: List[str]):
    """Step 7: Sequential Outlier Removal (Key to match Notebook)"""
    bounds_info = {}
    for col_name in columns:
        if col_name in df.columns:
            # TÍNH TOÁN LẠI NGƯỠNG SAU MỖI LẦN LỌC
            lower, upper = compute_iqr_bounds(df, col_name)
            bounds_info[col_name] = {"lower": lower, "upper": upper}
            df = df.filter((F.col(col_name) >= lower) & (F.col(col_name) <= upper))
    return df, bounds_info

def finalize_dataset(df):
    """Step 8: Final Clean and Target Transform"""
    # 9.2: Filter NULLs for ALL critical columns
    critical_cols = [
        'kWhDelivered', 'duration', 'charging_duration_log', 
        'lag_1_log', 'lag_2_log', 'lag_3_log',
        'rolling_mean_3_log', 'rolling_mean_5_log'
    ]
    for c in critical_cols:
        df = df.filter(F.col(c).isNotNull())
    
    # Filter negative (Important!)
    df = df.filter(F.col("charging_duration") >= 0)
    df = df.filter(F.col("duration") >= 0)

    # 9.4: Log Transform Target
    df = df.withColumn("kWhDelivered_log", F.log1p(F.col("kWhDelivered")))

    # 9.5: Feature Selection
    features_paper = [
        'hour', 'day_of_week', 'month', 'season',
        'duration', 'charging_duration', 'charging_duration_log',
        'hour_sin', 'hour_cos', 'day_of_year', 'week_of_year', 'is_holiday',
        'lag_1_log', 'lag_2_log', 'lag_3_log',
        'rolling_mean_3_log', 'rolling_mean_5_log'
    ]
    final_cols = features_paper + ['kWhDelivered_log', 'connectionTime_utc']
    return df.select(*[c for c in final_cols if c in df.columns])

def run_pipeline(df, config: Dict[str, Any] = None):
    """Main Orchestrator"""
    # Pre-processing
    df, _ = drop_id_columns(df)
    df = convert_time(df)
    df = add_duration_features(df)
    df = add_temporal_features(df)
    
    # Windowing (Lags/Rolling)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    
    # Outliers (Lọc cuốn chiếu để thu hẹp tập dữ liệu đúng như notebook)
    df, bounds = remove_outliers_iqr(df, ["duration", "charging_duration", "kWhDelivered"])
    
    # Final Clean & Target
    df = finalize_dataset(df)
    
    return df, {"outlier_bounds": bounds}