# spark-submit src/jobs/acn_scaling_data.py \
# hdfs://localhost:9000/ev-project/data/bronze/ev_sessions/caltech/*/*/* \
# hdfs://localhost:9000/data/benchmark/acn/1gb \
# 50 \
# 12 \
# metrics_scaling.csv


#!/bin/bash

# Chạy scaling lên 1GB (Scale Factor 50, 12 Partitions)
echo "Executing 1GB Scale Benchmark Job..."
spark-submit src/jobs/acn_scaling_data.py 25 12

# Gợi ý: Nếu sau này cần chạy thử nghiệm 500MB, bạn chỉ cần gọi:
# spark-submit src/jobs/acn_scaling_data.py 25 12