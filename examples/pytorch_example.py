import time
import torch

from ml_profiler.collectors.cpu_psutil import CPUCollector
from ml_profiler.collectors.gpu_nvml import NVMLCollector
from ml_profiler.exporters.json_exporter import JSONExporter
from ml_profiler.exporters.tensorboard_exporter import TensorBoardExporter
from ml_profiler.instrumentation.profiler_manager import ProfilerManager


device = "cuda" if torch.cuda.is_available() else "cpu"

collectors = [NVMLCollector()] if device == "cuda" else []
collectors.append(CPUCollector())

exporters = [
    JSONExporter("logs/torch_run.json"),
    TensorBoardExporter("runs/demo"),
]

# simple model + loop to exercise GPU
model = torch.nn.Sequential(
    torch.nn.Linear(1024, 2048),
    torch.nn.ReLU(),
    torch.nn.Linear(2048, 1024),
).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

with ProfilerManager(collectors, exporters, interval=0.25):
    for step in range(200):
        x = torch.randn(256, 1024, device=device)
        y = model(x)
        loss = (y ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 20 == 0:
            print(f"step={step} loss={loss.item():.4f}")

print("Done. JSON written to logs/torch_run.json; TensorBoard logs in runs/demo")
