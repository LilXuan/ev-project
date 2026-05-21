#!/bin/bash

export SPARK_HOME=/opt/spark

$SPARK_HOME/sbin/stop-worker.sh
$SPARK_HOME/sbin/stop-master.sh

echo "Cluster stopped."
