import time
from ml_profiler.collectors.cpu_psutil import CPUCollector
from ml_profiler.exporters.json_exporter import JSONExporter
from ml_profiler.instrumentation.profiler_manager import ProfilerManager


collectors = [CPUCollector()]
exporters = [JSONExporter("logs/cpu_only.json")]

with ProfilerManager(collectors, exporters, interval=0.25):
    for _ in range(40):
        # simulate work
        s = 0
        for i in range(1000000):
            s += i
        time.sleep(0.05)
print("Done. JSON written to logs/cpu_only.json")
