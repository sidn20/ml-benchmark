import subprocess
import threading
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

def run_with_stress(n_runs: int, n_samples: int, cpu_workers: int = 2):
    print(f"\nStarting stress-ng with {cpu_workers} CPU workers...")
    
    # Start stress-ng in background as a subprocess
    stress = subprocess.Popen(
        ["stress-ng", "--cpu", str(cpu_workers), "--timeout", "120s"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    print("Stress running! Waiting 2 seconds for load to build...\n")
    time.sleep(2)
    
    try:
        results = run_benchmark_suite(n_runs=n_runs)
        print_summary(results)
    finally:
        stress.terminate()
        stress.wait()
        print("Stress-ng stopped.")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML Benchmark Tool")
    parser.add_argument("--runs", type=int, default=5, help="Number of benchmark runs")
    parser.add_argument("--samples", type=int, default=1000, help="Dataset size per run")
    parser.add_argument("--report", action="store_true", help="Generate CSV and PNG report")
    parser.add_argument("--stress", action="store_true", help="Run under CPU stress load")
    parser.add_argument("--stress-workers", type=int, default=2, help="Number of stress CPU workers")
    parser.add_argument("--compare", action="store_true", help="Compare normal vs stressed runs")
    args = parser.parse_args()

    if args.compare:
        print("=== NORMAL RUN ===")
        normal = run_benchmark_suite(n_runs=args.runs)
        print_summary(normal)

        print("\n=== STRESSED RUN ===")
        stressed = run_with_stress(n_runs=args.runs, n_samples=args.samples, cpu_workers=args.stress_workers)

        # Compare the two
        normal_mean = statistics.mean([r.latency_ms for r in normal])
        stressed_mean = statistics.mean([r.latency_ms for r in stressed])
        diff = stressed_mean - normal_mean
        pct = (diff / normal_mean) * 100

        print(f"\n{'='*45}")
        print(f"  COMPARISON RESULT")
        print(f"{'='*45}")
        print(f"  Normal mean   : {normal_mean:.1f} ms")
        print(f"  Stressed mean : {stressed_mean:.1f} ms")
        print(f"  Difference    : +{diff:.1f} ms ({pct:.1f}% slower)")
        print(f"{'='*45}")

        if args.report:
            from reporter import save_csv, plot_results
            df = save_csv(normal + stressed)
            plot_results(df)

    elif args.stress:
        results = run_with_stress(n_runs=args.runs, n_samples=args.samples, cpu_workers=args.stress_workers)
        if args.report:
            from reporter import save_csv, plot_results
            df = save_csv(results)
            plot_results(df)
    else:
        results = run_benchmark_suite(n_runs=args.runs)
        print_summary(results)
        if args.report:
            from reporter import save_csv, plot_results
            df = save_csv(results)
            plot_results(df)
