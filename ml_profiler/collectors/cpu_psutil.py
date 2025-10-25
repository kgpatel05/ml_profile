import psutil
import time


class CPUCollector:
    """Collects host CPU and memory utilization via psutil."""

    def collect(self) -> dict:
        return {
            "timestamp": time.time(),
            "cpu.percent": float(psutil.cpu_percent(interval=None)),
            "mem.percent": float(psutil.virtual_memory().percent),
        }
