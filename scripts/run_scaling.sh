spark-submit src/jobs/acn_scaling_data.py \
hdfs://localhost:9000/ev-project/data/bronze/ev_sessions/caltech/*/*/* \
hdfs://localhost:9000/data/benchmark/acn/1gb \
50 \
12 \
metrics_scaling.csv