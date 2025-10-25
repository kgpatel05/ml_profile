"""Optional helper to use torch.profiler as a context manager.

Usage:
    from ml_profiler.collectors.pytorch_profiler import TorchProfiler
    with TorchProfiler(logdir="runs/torch_prof") as prof:
        for step, batch in enumerate(loader):
            ...
            prof.step()  # marks a profiling step

This is separate from ProfilerManager because torch.profiler already
handles its own data collection/export via TensorBoard trace files.
"""

from __future__ import annotations


try:  # torch is optional
    import torch
    from torch.profiler import profile, record_function, ProfilerActivity
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "TorchProfiler requires PyTorch with profiler support installed."
    ) from e


class TorchProfiler:
    def __init__(self, logdir: str = "runs/torch_prof", use_cuda: bool = True):
        acts = [ProfilerActivity.CPU]
        if use_cuda and torch.cuda.is_available():
            acts.append(ProfilerActivity.CUDA)
        self.prof = profile(
            activities=acts,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(logdir),
            record_shapes=True,
            with_stack=True,
            profile_memory=True,
        )

    def __enter__(self):
        return self.prof.__enter__()

    def __exit__(self, exc_type, exc, tb):
        return self.prof.__exit__(exc_type, exc, tb)

    def step(self):
        self.prof.step()
