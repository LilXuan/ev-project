# Thêm thư mục gốc vào PYTHONPATH để Python tìm thấy folder configs/
export PYTHONPATH=$PYTHONPATH:$(pwd)

cd /home/axuan/bigdata/ev-project
# Thêm thư mục configs vào PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:${PWD}"

# Đảm bảo thư mục logs tồn tại
mkdir -p metrics

#### current use================================

spark-submit \
  --master spark://192.168.40.130:7077 \
  --deploy-mode client \
  --name "ACN_Processing" \
  --num-executors 2 \
  --executor-cores 1 \
  --executor-memory 1g \
  --driver-memory 1g \
  \
  --conf "spark.pyspark.python=python3" \
  --conf "spark.pyspark.driver.python=python3" \
  \
  --conf "spark.driver.extraJavaOptions=-javaagent:/opt/jmx_exporter/jmx_prometheus_javaagent-1.5.0.jar=7071:/opt/jmx_exporter/config.yaml" \
  --conf "spark.executor.extraJavaOptions=-javaagent:/opt/jmx_exporter/jmx_prometheus_javaagent-1.5.0.jar=7072:/opt/jmx_exporter/config.yaml" \
  \
  --conf "spark.ui.prometheus.enabled=true" \
  \
  --conf "spark.metrics.conf.*.sink.prometheusServlet.class=org.apache.spark.metrics.sink.PrometheusServlet" \
  --conf "spark.metrics.conf.*.sink.prometheusServlet.path=/metrics/prometheus" \
  \
  --conf "spark.metrics.conf.*.sink.jmx.class=org.apache.spark.metrics.sink.JmxSink" \
  \
  --conf "spark.metrics.conf.*.source.jvm.class=org.apache.spark.metrics.source.JvmSource" \
  --conf "spark.metrics.conf.*.source.DAGSchedulerSource.class=org.apache.spark.metrics.source.DAGSchedulerSource" \
  --conf "spark.metrics.conf.*.source.BlockManagerSource.class=org.apache.spark.metrics.source.BlockManagerSource" \
  --conf "spark.metrics.conf.*.source.ExecutorSource.class=org.apache.spark.metrics.source.ExecutorSource" \
  \
  src/jobs/acn_processing_data.py configs.config_processing