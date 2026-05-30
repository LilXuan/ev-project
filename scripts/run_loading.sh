#!/bin/bash

# Thêm thư mục gốc vào PYTHONPATH để Python tìm thấy folder configs/
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "=================================="
echo "Experiment 1 - 19MB"
echo "=================================="

spark-submit \
src/jobs/acn_loading_data.py \
configs.exp1_19mb

# echo "=================================="
# echo "Experiment 2 - 100MB"
# echo "=================================="

# spark-submit \
# src/jobs/acn_loading_data.py \
# configs.exp2_100mb

# echo "=================================="
# echo "Experiment 3 - 1GB"
# echo "=================================="

# spark-submit \
# src/jobs/acn_loading_data.py \
# configs.exp3_1gb

# echo "All experiments completed."