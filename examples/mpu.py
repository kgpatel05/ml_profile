### `examples/pytorch_mps_example.py`

"""PyTorch + MPS example for Apple Silicon (no CUDA required)."""
import torch
from ml_profiler.collectors.cpu_psutil import CPUCollector
from ml_profiler.collectors.mps_torch import MPSCollector
from ml_profiler.exporters.json_exporter import JSONExporter
from ml_profiler.exporters.tensorboard_exporter import TensorBoardExporter
from ml_profiler.instrumentation.profiler_manager import ProfilerManager

use_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
device = torch.device('mps' if use_mps else 'cpu')
print(f'Using device: {device}')

collectors = [CPUCollector()]
if use_mps:
    collectors.insert(0, MPSCollector())

exporters = [
    JSONExporter('logs/mps_run.json'),
    TensorBoardExporter('runs/mps_demo'),
]

model = torch.nn.Sequential(
    torch.nn.Linear(512, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 512),
).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

with ProfilerManager(collectors, exporters, interval=0.25):
    for step in range(100):
        x = torch.randn(128, 512, device=device)
        y = model(x)
        loss = (y ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        # 🩵 fix Metal buffer crash
        if use_mps:
            torch.mps.synchronize()

        if step % 10 == 0:
            print(f"step={step}, loss={loss.item():.4f}")


print('✅ Done. JSON written to logs/mps_run.json; TensorBoard logs in runs/mps_demo')