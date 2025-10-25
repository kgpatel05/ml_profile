"""MPS (Apple Silicon / Metal) metrics via PyTorch."""

import time
import torch


class MPSCollector:
    """Collects MPS memory allocation info on Apple Silicon."""

    def __init__(self):
        self._has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    def _get_val(self, fn_name):
        fn = getattr(torch.mps, fn_name, None)
        if callable(fn):
            try:
                return float(fn()) / (1024 ** 2)  # MB
            except Exception:
                return None
        return None

    def collect(self):
        out = {"timestamp": time.time(), "mps.available": 1.0 if self._has_mps else 0.0}
        if not self._has_mps:
            return out
        try:
            torch.mps.synchronize()
        except Exception:
            pass
        for name in [
            "current_allocated_memory",
            "driver_allocated_memory",
            "current_reserved_memory",
            "driver_reserved_memory",
        ]:
            val = self._get_val(name)
            if val is not None:
                out[f"mps.{name}_mb"] = val
        return out
