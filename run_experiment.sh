#!/bin/bash

# Thêm thư mục gốc vào PYTHONPATH để Python tìm thấy folder configs/
export PYTHONPATH=$PYTHONPATH:$(pwd)

# echo "--- Đang chạy Thực nghiệm 1: 31k dòng ---"
# spark-submit src/jobs/acn_loading_data.py configs.exp1_31k

# echo "--- Đang chạy Thực nghiệm 2: 500k dòng ---"
# Giả sử bạn đã tạo file configs/exp2_500k.py
# spark-submit src/jobs/acn_loading_data.py configs.exp2_500k

# echo "--- Đang chạy Thực nghiệm 1: 31 dòng preprocessing step 2 3---"

# # Or with spark-submit in local mode
# spark-submit \
#   --master local[*] \
#   --driver-memory 4g \
#   src/jobs/acn_processing_data.py config_caltech
#!/bin/bash

#!/bin/bash

cd /home/axuan/bigdata/ev-project

# Thêm thư mục configs vào PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:${PWD}"

# # Chạy job với config đầy đủ đường dẫn
# spark-submit \
#   --master local[*] \
#   --driver-memory 2g \
#   --conf spark.pyspark.python=python3 \
#   src/jobs/acn_processing_data.py configs.config_caltech

# Chạy job với config đầy đủ đường dẫn
spark-submit \
  --master local[*] \
  --driver-memory 2g \
  --conf spark.pyspark.python=python3 \
  src/jobs/acn_training_model.py configs.config_caltech