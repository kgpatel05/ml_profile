"""Optional Prometheus exporter that exposes gauges for each metric key.
Install with: pip install 'prometheus-client'
Add to exporters list to enable live scraping at :8000/metrics.
"""
from __future__ import annotations
from typing import Dict


try:
    from prometheus_client import Gauge, start_http_server
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Prometheus exporter requires 'prometheus-client'. Install with extras: [dashboard]."
    ) from e


class PrometheusExporter:
    def __init__(self, port: int = 8000):
        self.gauges: Dict[str, Gauge] = {}
        start_http_server(port)
        print(f"📈 Prometheus metrics available at http://localhost:{port}/metrics")

    def _g(self, name: str) -> Gauge:
        key = name.replace(".", ":")  # Prometheus-friendly
        if key not in self.gauges:
            self.gauges[key] = Gauge(key, f"ml_profiler metric {name}")
        return self.gauges[key]

    def on_sample(self, sample: dict, idx: int):
        for k, v in sample.items():
            if k == "timestamp":
                continue
            if isinstance(v, (int, float)):
                self._g(k).set(v)
