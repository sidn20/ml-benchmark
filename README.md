# ML Benchmark Tool

A command-line tool that benchmarks ML model inference performance while capturing live Linux system metrics (CPU, RAM, frequency).

Built with Python, psutil, scikit-learn, pandas, and matplotlib.

## What it measures
- Inference latency per run (ms)
- Min / Max / Mean / Stdev / p95 latency
- CPU usage and RAM consumption during each run

## Project Structure
- `collector.py` — reads live Linux system metrics using psutil
- `benchmark.py` — runs ML workload and measures latency
- `reporter.py` — saves results to CSV and generates charts

## Usage

```bash
# Basic run
python3 benchmark.py --runs 5

# With report output
python3 benchmark.py --runs 10 --report

# Larger dataset
python3 benchmark.py --runs 10 --samples 5000 --report
```

## Sample Output
- `results.csv` — per-run latency and system metrics
- `report.png` — latency, CPU, and RAM charts

## Requirements
```bash
pip3 install transformers psutil pandas matplotlib scikit-learn
```
