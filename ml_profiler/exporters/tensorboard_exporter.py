from __future__ import annotations
from typing import Iterable


# Prefer torch SummaryWriter if available; fall back to tensorboardX
try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:  # pragma: no cover
    try:
        from tensorboardX import SummaryWriter  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "TensorBoard exporter requires torch.utils.tensorboard or tensorboardX installed."
        ) from e


class TensorBoardExporter:
    def __init__(self, logdir: str = "runs/ml_profiler"):
        self.writer = SummaryWriter(logdir)

    def on_sample(self, sample: dict, idx: int):
        for k, v in sample.items():
            if k == "timestamp":
                continue
            if isinstance(v, (int, float)):
                self.writer.add_scalar(k, v, idx)

    def export(self, data: Iterable[dict]):
        # no-op here; we streamed as we went
        self.writer.close()
        print(f"✅ TensorBoard logs written to {self.writer.log_dir}")
