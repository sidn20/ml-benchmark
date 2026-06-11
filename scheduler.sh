#!/bin/bash

# Config
BENCHMARK_DIR="$HOME/ml_benchmark"
LOG_FILE="$BENCHMARK_DIR/benchmark_log.csv"
THRESHOLD_MS=200
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Create log file with header if it doesn't exist
if [ ! -f "$LOG_FILE" ]; then
    echo "timestamp,min_ms,max_ms,mean_ms,stdev_ms" > "$LOG_FILE"
    echo "Log file created at $LOG_FILE"
fi

echo "[$TIMESTAMP] Running benchmark..."

# Run benchmark and capture output
OUTPUT=$(python3 "$BENCHMARK_DIR/benchmark.py" --runs 5)

# Parse results using grep and awk
MIN=$(echo "$OUTPUT"   | grep "Min latency"  | awk '{print $(NF-1)}')
MAX=$(echo "$OUTPUT"   | grep "Max latency"  | awk '{print $(NF-1)}')
MEAN=$(echo "$OUTPUT"  | grep "Mean latency" | awk '{print $(NF-1)}')
STDEV=$(echo "$OUTPUT" | grep "Stdev"        | awk '{print $(NF-1)}')
# Log to CSV
echo "$TIMESTAMP,$MIN,$MAX,$MEAN,$STDEV" >> "$LOG_FILE"
echo "[$TIMESTAMP] Logged — mean: ${MEAN}ms, max: ${MAX}ms"

# Check threshold
MEAN_INT=$(echo "$MEAN" | cut -d'.' -f1)
if [ "$MEAN_INT" -gt "$THRESHOLD_MS" ]; then
    echo "[$TIMESTAMP] ALERT: Mean latency ${MEAN}ms exceeds threshold ${THRESHOLD_MS}ms" | tee -a "$BENCHMARK_DIR/alerts.log"
    bash "$BENCHMARK_DIR/alert.sh" "$MEAN" "$THRESHOLD_MS"
fi
