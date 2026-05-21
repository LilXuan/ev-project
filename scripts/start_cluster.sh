#!/bin/bash

export SPARK_HOME=/opt/spark

echo "Starting Spark Master..."
$SPARK_HOME/sbin/start-master.sh

sleep 3

echo "Starting Worker 1..."
SPARK_WORKER_CORES=1 
SPARK_WORKER_MEMORY=1g 
SPARK_WORKER_PORT=3381 
SPARK_WORKER_WEBUI_PORT=8081 
$SPARK_HOME/sbin/start-worker.sh spark://192.168.40.130:7077

echo "Starting Worker 2..."
SPARK_WORKER_CORES=1 
SPARK_WORKER_MEMORY=1g 
SPARK_WORKER_PORT=3382 
SPARK_WORKER_WEBUI_PORT=8082 
$SPARK_HOME/sbin/start-worker.sh spark://192.168.40.130:7077

echo "Cluster started."
