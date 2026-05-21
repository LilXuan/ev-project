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
#   --master "local-cluster[2,1,2048]" \
#   --driver-memory 1g \
#   --executor-memory 1g \
#   --conf spark.executor.instances=1 \
#   --conf spark.pyspark.python=python3 \
#   --conf spark.driver.extraJavaOptions=javaagent \
#   --conf "spark.metrics.conf=/opt/spark/conf/metrics.properties" \
#   --conf "spark.driver.extraJavaOptions=-javaagent:/opt/jmx_exporter/jmx_prometheus_javaagent-1.5.0.jar=7071:/opt/jmx_exporter/config.yaml" \
#   --conf "spark.executor.extraJavaOptions=-javaagent:/opt/jmx_exporter/jmx_prometheus_javaagent-1.5.0.jar=7072:/opt/jmx_exporter/config.yaml" \
#   src/jobs/acn_processing_data.py configs.config_caltech


# Đảm bảo thư mục logs tồn tại
mkdir -p metrics

# spark-submit \
#   --master local-cluster[2,1,1024] \
#   --name "ACN_Processing_Debug" \
#   --driver-memory 2g \
#   --conf "spark.pyspark.python=python3" \
#   --conf "spark.pyspark.driver.python=python3" \
#   --conf "spark.driver.extraJavaOptions=-javaagent:/opt/jmx_exporter/jmx_prometheus_javaagent-1.5.0.jar=7071:/opt/jmx_exporter/config.yaml" \
#   --conf "spark.executor.extraJavaOptions=-javaagent:/opt/jmx_exporter/jmx_prometheus_javaagent-1.5.0.jar=7072:/opt/jmx_exporter/config.yaml" \
#   --conf "spark.ui.prometheus.enabled=true" \
#   --conf "spark.metrics.conf.*.sink.prometheusServlet.class=org.apache.spark.metrics.sink.PrometheusServlet" \
#   --conf "spark.metrics.conf.*.sink.prometheusServlet.path=/metrics/prometheus" \
#   --conf "spark.metrics.conf.*.sink.jmx.class=org.apache.spark.metrics.sink.JmxSink" \
#   --conf "spark.metrics.conf.*.source.jvm.class=org.apache.spark.metrics.source.JvmSource" \
#   --conf "spark.metrics.conf.*.source.DAGSchedulerSource.class=org.apache.spark.metrics.source.DAGSchedulerSource" \
#   --conf "spark.metrics.conf.*.source.BlockManagerSource.class=org.apache.spark.metrics.source.BlockManagerSource" \
#   --conf "spark.metrics.conf.*.source.ExecutorSource.class=org.apache.spark.metrics.source.ExecutorSource" \
#   src/jobs/acn_processing_data.py configs.config_caltech


# # oke nhưng  chua hoan chinh vi thei executor metrics
# spark-submit \
#   --master spark://192.168.40.130:7077 \
#   --deploy-mode client \
#   --name "ACN_Processing" \
#   --driver-memory 2g \
#   --conf "spark.ui.prometheus.enabled=true" \
#   --conf "spark.metrics.conf.*.sink.prometheusServlet.class=org.apache.spark.metrics.sink.PrometheusServlet" \
#   --conf "spark.metrics.conf.*.sink.prometheusServlet.path=/metrics/prometheus" \
#   src/jobs/acn_processing_data.py configs.config_caltech


#### current use================================

# spark-submit \
#   --master spark://192.168.40.130:7077 \
#   --deploy-mode client \
#   --name "ACN_Processing" \
#   --num-executors 2 \
#   --executor-cores 1 \
#   --executor-memory 1g \
#   --driver-memory 1g \
#   \
#   --conf "spark.pyspark.python=python3" \
#   --conf "spark.pyspark.driver.python=python3" \
#   \
#   --conf "spark.driver.extraJavaOptions=-javaagent:/opt/jmx_exporter/jmx_prometheus_javaagent-1.5.0.jar=7071:/opt/jmx_exporter/config.yaml" \
#   --conf "spark.executor.extraJavaOptions=-javaagent:/opt/jmx_exporter/jmx_prometheus_javaagent-1.5.0.jar=7072:/opt/jmx_exporter/config.yaml" \
#   \
#   --conf "spark.ui.prometheus.enabled=true" \
#   \
#   --conf "spark.metrics.conf.*.sink.prometheusServlet.class=org.apache.spark.metrics.sink.PrometheusServlet" \
#   --conf "spark.metrics.conf.*.sink.prometheusServlet.path=/metrics/prometheus" \
#   \
#   --conf "spark.metrics.conf.*.sink.jmx.class=org.apache.spark.metrics.sink.JmxSink" \
#   \
#   --conf "spark.metrics.conf.*.source.jvm.class=org.apache.spark.metrics.source.JvmSource" \
#   --conf "spark.metrics.conf.*.source.DAGSchedulerSource.class=org.apache.spark.metrics.source.DAGSchedulerSource" \
#   --conf "spark.metrics.conf.*.source.BlockManagerSource.class=org.apache.spark.metrics.source.BlockManagerSource" \
#   --conf "spark.metrics.conf.*.source.ExecutorSource.class=org.apache.spark.metrics.source.ExecutorSource" \
#   \
#   src/jobs/acn_processing_data.py configs.config_caltech






# ================ TRAINING MODEL =======================



# Chạy job với config đầy đủ đường dẫn
# spark-submit \
#   --master local[*] \
#   --driver-memory 2g \
#   --conf spark.pyspark.python=python3 \
#   --conf "spark.ui.enabled=true" \
#   src/jobs/acn_training_model.py configs.config_caltech


spark-submit \
  --master spark://192.168.40.130:7077 \
  --deploy-mode client \
  --name "ACN_Model_Training" \
  \
  --num-executors 2 \
  --executor-cores 1 \
  --executor-memory 1g \
  --driver-memory 2g \
  \
  --conf "spark.pyspark.python=python3" \
  --conf "spark.pyspark.driver.python=python3" \
  \
  --conf "spark.ui.prometheus.enabled=true" \
  \
  --conf "spark.metrics.conf.*.sink.prometheusServlet.class=org.apache.spark.metrics.sink.PrometheusServlet" \
  --conf "spark.metrics.conf.*.sink.prometheusServlet.path=/metrics/prometheus" \
  \
  --conf "spark.eventLog.enabled=true" \
  --conf "spark.eventLog.dir=hdfs://192.168.40.130:9000/spark-logs" \
  \
  --conf "spark.sql.shuffle.partitions=4" \
  \
  src/jobs/acn_training_model.py configs.config_caltech