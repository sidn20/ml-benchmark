import pandas as pd
import matplotlib
matplotlib.use('Agg')  # no display needed, saves to file
import matplotlib.pyplot as plt
from benchmark import BenchmarkResult

def save_csv(results: list[BenchmarkResult], path: str = "results.csv"):
    rows = []
    for r in results:
        avg_cpu = sum(s.cpu_percent for s in r.system_snapshots) / len(r.system_snapshots) if r.system_snapshots else 0
        avg_ram = sum(s.ram_used_mb for s in r.system_snapshots) / len(r.system_snapshots) if r.system_snapshots else 0
        rows.append({
            "run_id": r.run_id,
            "latency_ms": round(r.latency_ms, 2),
            "avg_cpu_percent": round(avg_cpu, 2),
            "avg_ram_mb": round(avg_ram, 2)
        })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"Results saved to {path}")
    return df

def plot_results(df: pd.DataFrame, path: str = "report.png"):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("ML Benchmark Report", fontsize=14, fontweight='bold')

    axes[0].bar(df["run_id"], df["latency_ms"], color="steelblue")
    axes[0].set_title("Latency per Run")
    axes[0].set_xlabel("Run")
    axes[0].set_ylabel("Latency (ms)")

    axes[1].plot(df["run_id"], df["avg_cpu_percent"], marker='o', color="tomato")
    axes[1].set_title("Avg CPU % per Run")
    axes[1].set_xlabel("Run")
    axes[1].set_ylabel("CPU %")

    axes[2].plot(df["run_id"], df["avg_ram_mb"], marker='s', color="seagreen")
    axes[2].set_title("Avg RAM Usage per Run")
    axes[2].set_xlabel("Run")
    axes[2].set_ylabel("RAM (MB)")

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"Chart saved to {path}")

if __name__ == "__main__":
    from benchmark import run_benchmark_suite
    results = run_benchmark_suite(n_runs=5)
    df = save_csv(results)
    plot_results(df)
    print("\nFinal DataFrame:")
    print(df.to_string(index=False))
