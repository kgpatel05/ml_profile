from __future__ import annotations
import time


try:
    import pynvml  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "pynvml is required for NVMLCollector. Install on systems with NVIDIA GPUs/drivers."
    ) from e


class NVMLCollector:
    """Collects per-GPU utilization and memory via NVML.

    Gracefully handles multi-GPU machines by reporting metrics per index:
    - gpu{idx}.util
    - gpu{idx}.mem.used_mb
    - gpu{idx}.mem.total_mb
    - gpu{idx}.power_w (if available)
    - gpu{idx}.temp_c (if available)
    """

    def __init__(self, device_indices: list[int] | None = None):
        pynvml.nvmlInit()
        self._initialized = True
        self.device_count = pynvml.nvmlDeviceGetCount()
        if device_indices is None:
            self.device_indices = list(range(self.device_count))
        else:
            self.device_indices = [i for i in device_indices if 0 <= i < self.device_count]

    def collect(self) -> dict:
        ts = time.time()
        out: dict[str, float] = {"timestamp": ts}
        for idx in self.device_indices:
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            out[f"gpu{idx}.util"] = float(util.gpu)
            out[f"gpu{idx}.mem.used_mb"] = mem.used / (1024 ** 2)
            out[f"gpu{idx}.mem.total_mb"] = mem.total / (1024 ** 2)
            # Optional metrics
            try:
                p = pynvml.nvmlDeviceGetPowerUsage(handle)  # milliwatts
                out[f"gpu{idx}.power_w"] = p / 1000.0
            except Exception:
                pass
            try:
                t = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                out[f"gpu{idx}.temp_c"] = float(t)
            except Exception:
                pass
        return out

    def shutdown(self):  # optional explicit shutdown
        if getattr(self, "_initialized", False):
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._initialized = False
