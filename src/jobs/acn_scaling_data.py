# # src/jobs/acn_scaling_data.py

import sys
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    rand, lit, explode, sequence,
    col, expr, concat
)

def run_scaling(input_path, output_path, scale_factor, partitions, metrics_log):

    spark = SparkSession.builder \
        .appName(f"scaling_exp_{scale_factor}") \
        .config("spark.executor.memory", "2g") \
        .config("spark.driver.memory", "1g") \
        .getOrCreate()

    try:
        # =======================
        # 1. READ
        # =======================
        start_read = time.time()

        if "parquet" in input_path:
            df = spark.read.parquet(input_path)
        else:
            df = spark.read.json(input_path)

        input_rows = df.count()
        read_time = time.time() - start_read
        input_partitions = df.rdd.getNumPartitions()

        # =======================
        # 2. REPARTITION + CACHE
        # =======================
        df = df.repartition(partitions).cache()
        df.count()  # materialize

        # =======================
        # 3. SCALING (REPLICATE)
        # =======================
        start_scale = time.time()

        df_big = df.withColumn(
            "batch_id",
            explode(sequence(lit(0), lit(scale_factor - 1)))
        )

        # =======================
        # 4. PERTURB DATA (🔥 CORE)
        # =======================

        # --- ID uniqueness (tránh duplicate key)
        df_big = df_big.withColumn(
            "userID",
            concat(col("userID"), lit("_"), col("batch_id"))
        ).withColumn(
            "sessionID",
            concat(col("sessionID"), lit("_"), col("batch_id"))
        )

        # --- numeric variation (giữ distribution nhưng không identical)
        df_big = df_big.withColumn(
            "kWhDelivered",
            col("kWhDelivered") * (1 + (rand() - 0.5) * 0.2)  # ±10%
        )

        # --- time shift (rất quan trọng cho ML)
        df_big = df_big.withColumn(
            "connectionTime",
            expr("connectionTime")  # giữ nguyên nếu string
        )

        # nếu bạn parse được timestamp thì dùng cái này tốt hơn:
        # df_big = df_big.withColumn(
        #     "connectionTime",
        #     expr("timestampadd(DAY, batch_id % 30, connectionTime)")
        # )

        # --- thêm noise column
        df_big = df_big.withColumn("noise", rand())

        # =======================
        # 5. ACTION (FOR TIMING)
        # =======================
        output_rows = df_big.count()
        scaling_time = time.time() - start_scale

        # =======================
        # 6. REPARTITION OUTPUT
        # =======================
        df_big = df_big.repartition(partitions * 2)
        output_partitions = df_big.rdd.getNumPartitions()

        # =======================
        # 7. WRITE (OVERWRITE)
        # =======================
        start_write = time.time()

        df_big.write \
            .mode("overwrite") \
            .option("compression", "snappy") \
            .parquet(output_path)

        write_time = time.time() - start_write

        # =======================
        # 8. METRICS
        # =======================
        rows_per_sec = output_rows / scaling_time

        print(f"""
[SCALING RESULT - EXPERIMENT]
Input Rows: {input_rows}
Output Rows: {output_rows}
Scale Factor: {scale_factor}

Read Time: {read_time:.2f}s
Scaling Time: {scaling_time:.2f}s
Write Time: {write_time:.2f}s

Throughput: {rows_per_sec:.2f} rows/sec

Input Partitions: {input_partitions}
Output Partitions: {output_partitions}
""")

        with open(metrics_log, "a") as f:
            f.write(
                f"{scale_factor},{input_rows},{output_rows},"
                f"{read_time},{scaling_time},{write_time},"
                f"{rows_per_sec},{input_partitions},{output_partitions}\n"
            )

    finally:
        spark.stop()


# =======================
# MAIN
# =======================
if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("Usage: <input> <output> <scale_factor> <partitions> <metrics_log>")
        sys.exit(1)

    run_scaling(
        sys.argv[1],
        sys.argv[2],
        int(sys.argv[3]),
        int(sys.argv[4]),
        sys.argv[5]
    )