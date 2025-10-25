from __future__ import annotations
import threading
import time
from typing import List


class ProfilerManager:
    """Orchestrates multiple collectors and exporters.

    Runs collectors in a background thread at a fixed interval,
    accumulates samples in memory, then exports on stop.
    """

    def __init__(self, collectors: List, exporters: List, interval: float = 0.5):
        self.collectors = collectors
        self.exporters = exporters
        self.interval = interval
        self.data: List[dict] = []
        self.running = False
        self.thread = None

    def _collect_once(self) -> dict:
        merged = {}
        for c in self.collectors:
            try:
                d = c.collect()
                if not isinstance(d, dict):
                    continue
                # merge; later collectors can override
                merged.update(d)
            except Exception as e:
                # be resilient; skip faulty collector for this tick
                merged.setdefault("collector_errors", 0)
                merged["collector_errors"] += 1
        return merged

    def _loop(self):
        idx = 0
        while self.running:
            sample = self._collect_once()
            self.data.append(sample)
            # live push to exporters that support streaming
            for e in self.exporters:
                fn = getattr(e, "on_sample", None)
                if callable(fn):
                    try:
                        fn(sample, idx)
                    except Exception:
                        pass
            idx += 1
            time.sleep(self.interval)

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def stop(self):
        if not self.running:
            return
        self.running = False
        if self.thread is not None:
            self.thread.join()
            self.thread = None
        for e in self.exporters:
            try:
                e.export(self.data)
            except Exception:
                pass

    # context manager sugar
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()
