#!/bin/bash

MEAN=$1
THRESHOLD=$2
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
ALERT_FILE="$HOME/ml_benchmark/alerts.log"

echo "======================================" >> "$ALERT_FILE"
echo "ALERT: $TIMESTAMP" >> "$ALERT_FILE"
echo "Mean latency: ${MEAN}ms" >> "$ALERT_FILE"
echo "Threshold   : ${THRESHOLD}ms" >> "$ALERT_FILE"
echo "Action      : Investigate system load" >> "$ALERT_FILE"
echo "======================================" >> "$ALERT_FILE"

# Print to terminal in red
echo -e "\e[31m[ALERT] Latency ${MEAN}ms exceeded threshold ${THRESHOLD}ms — check alerts.log\e[0m"
