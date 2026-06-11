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
    # detect if we have both normal and stressed runs (more than 5 rows = comparison)
    is_comparison = len(df) > 5

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("ML Benchmark Report", fontsize=14, fontweight='bold')

    if is_comparison:
        mid = len(df) // 2
        normal_df = df.iloc[:mid]
        stressed_df = df.iloc[mid:]

        # Latency comparison
        axes[0].plot(normal_df["run_id"].values, normal_df["latency_ms"].values,
                     marker='o', label="Normal", color="steelblue")
        axes[0].plot(stressed_df["run_id"].values, stressed_df["latency_ms"].values,
                     marker='o', label="Stressed", color="tomato")
        axes[0].legend()
        axes[0].set_title("Latency: Normal vs Stressed")

        # CPU comparison
        axes[1].plot(normal_df["run_id"].values, normal_df["avg_cpu_percent"].values,
                     marker='o', label="Normal", color="steelblue")
        axes[1].plot(stressed_df["run_id"].values, stressed_df["avg_cpu_percent"].values,
                     marker='o', label="Stressed", color="tomato")
        axes[1].legend()
        axes[1].set_title("CPU %: Normal vs Stressed")

        # Bar chart summary
        means = [normal_df["latency_ms"].mean(), stressed_df["latency_ms"].mean()]
        bars = axes[2].bar(["Normal", "Stressed"], means, color=["steelblue", "tomato"])
        axes[2].set_title("Mean Latency Comparison")
        axes[2].set_ylabel("Latency (ms)")
        for bar, val in zip(bars, means):
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                        f"{val:.1f}ms", ha='center', fontweight='bold')
    else:
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
def save_model_report(normal: list, stressed: list, path: str = "model_report.png"):
    import numpy as np

    names = [r.model_name for r in normal]
    normal_latencies = [r.latency_ms for r in normal]
    stressed_latencies = [r.latency_ms for r in stressed]
    accuracies = [r.accuracy for r in normal]

    x = np.arange(len(names))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Model Comparison: Normal vs Stressed", fontsize=14, fontweight='bold')

    # Latency chart
    bars1 = axes[0].bar(x - width/2, normal_latencies, width, label="Normal", color="steelblue")
    bars2 = axes[0].bar(x + width/2, stressed_latencies, width, label="Stressed", color="tomato")
    axes[0].set_title("Latency by Model")
    axes[0].set_ylabel("Latency (ms)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=15)
    axes[0].legend()
    for bar in bars1:
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f"{bar.get_height():.0f}", ha='center', fontsize=8)
    for bar in bars2:
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f"{bar.get_height():.0f}", ha='center', fontsize=8)

    # Accuracy chart
    bars3 = axes[1].bar(names, accuracies, color="seagreen")
    axes[1].set_title("Accuracy by Model (Normal)")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_ylim(80, 100)
    axes[1].tick_params(axis='x', rotation=15)
    for bar, val in zip(bars3, accuracies):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f"{val:.1f}%", ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"Model report saved to {path}")


if __name__ == "__main__":
    from benchmark import run_benchmark_suite
    results = run_benchmark_suite(n_runs=5)
    df = save_csv(results)
    plot_results(df)
    print("\nFinal DataFrame:")
    print(df.to_string(index=False))
