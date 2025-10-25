from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Iterable


class JSONExporter:
    def __init__(self, filepath: str = "logs/profile.json"):
        self.filepath = filepath
        Path(os.path.dirname(filepath) or ".").mkdir(parents=True, exist_ok=True)

    def export(self, data: Iterable[dict]):
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(list(data), f, indent=2)
        print(f"✅ Exported profiling data to {self.filepath}")
