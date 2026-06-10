import time
import statistics
from dataclasses import dataclass, field
from collector import collect_metrics, SystemSnapshot
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
import numpy as np

@dataclass
class BenchmarkResult:
    run_id: int
    latency_ms: float
    system_snapshots: list[SystemSnapshot] = field(default_factory=list)

def simulate_workload(n_samples: int = 1000) -> tuple:
    X, y = make_classification(
        n_samples=n_samples,
        n_features=20,
        random_state=42
    )
    return X, y

def run_single_benchmark(run_id: int, X, y) -> BenchmarkResult:
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    
    # Start collecting metrics in background
    start = time.perf_counter()
    model.fit(X, y)
    predictions = model.predict(X)
    end = time.perf_counter()
    
    latency_ms = (end - start) * 1000
    snapshots = collect_metrics(duration_seconds=1)
    
    return BenchmarkResult(
        run_id=run_id,
        latency_ms=latency_ms,
        system_snapshots=snapshots
    )

def run_benchmark_suite(n_runs: int = 5) -> list[BenchmarkResult]:
    print(f"Running benchmark suite: {n_runs} runs\n")
    X, y = simulate_workload()
    results = []
    
    for i in range(n_runs):
        print(f"  Run {i+1}/{n_runs}...", end=" ", flush=True)
        result = run_single_benchmark(i+1, X, y)
        results.append(result)
        print(f"done — {result.latency_ms:.1f} ms")
    
    return results

def print_summary(results: list[BenchmarkResult]):
    latencies = [r.latency_ms for r in results]
    
    print(f"\n{'='*45}")
    print(f"  BENCHMARK SUMMARY ({len(results)} runs)")
    print(f"{'='*45}")
    print(f"  Min latency    : {min(latencies):.1f} ms")
    print(f"  Max latency    : {max(latencies):.1f} ms")
    print(f"  Mean latency   : {statistics.mean(latencies):.1f} ms")
    print(f"  Stdev          : {statistics.stdev(latencies):.1f} ms")
    print(f"  p95 latency    : {sorted(latencies)[int(len(latencies)*0.95)]:.1f} ms")
    print(f"{'='*45}\n")

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML Benchmark Tool")
    parser.add_argument("--runs", type=int, default=5, help="Number of benchmark runs")
    parser.add_argument("--samples", type=int, default=1000, help="Dataset size per run")
    parser.add_argument("--report", action="store_true", help="Generate CSV and PNG report")
    args = parser.parse_args()

    results = run_benchmark_suite(n_runs=args.runs)
    print_summary(results)

    if args.report:
        from reporter import save_csv, plot_results
        df = save_csv(results)
        plot_results(df)
        print("Report generated: results.csv and report.png")
