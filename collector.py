import psutil
import time
from dataclasses import dataclass

@dataclass
class SystemSnapshot:
    timestamp: float
    cpu_percent: float
    ram_used_mb: float
    ram_total_mb: float
    cpu_freq_mhz: float

def take_snapshot() -> SystemSnapshot:
    freq = psutil.cpu_freq()
    ram = psutil.virtual_memory()
    
    return SystemSnapshot(
        timestamp=time.time(),
        cpu_percent=psutil.cpu_percent(interval=0.1),
        ram_used_mb=ram.used / 1024 / 1024,
        ram_total_mb=ram.total / 1024 / 1024,
        cpu_freq_mhz=freq.current if freq else 0.0
    )

def collect_metrics(duration_seconds: int = 5) -> list[SystemSnapshot]:
    snapshots = []
    end_time = time.time() + duration_seconds
    
    while time.time() < end_time:
        snapshots.append(take_snapshot())
        time.sleep(0.5)
    
    return snapshots

if __name__ == "__main__":
    print("Collecting system metrics for 5 seconds...\n")
    results = collect_metrics(duration_seconds=5)
    
    for s in results:
        print(f"CPU: {s.cpu_percent:5.1f}%  |  RAM: {s.ram_used_mb:.0f}/{s.ram_total_mb:.0f} MB  |  Freq: {s.cpu_freq_mhz:.0f} MHz")
