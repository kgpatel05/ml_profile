import argparse
import subprocess
import time
from pathlib import Path

from ml_profiler.collectors.cpu_psutil import CPUCollector
from ml_profiler.exporters.json_exporter import JSONExporter
from ml_profiler.exporters.tensorboard_exporter import TensorBoardExporter
from ml_profiler.instrumentation.profiler_manager import ProfilerManager


def _build_collectors(include_gpu=True):
    collectors = [CPUCollector()]
    if include_gpu:
        try:
            from ml_profiler.collectors.gpu_nvml import NVMLCollector
            collectors.append(NVMLCollector())
        except Exception:
            pass
    return collectors


def _build_exporters(json_path=None, tb_path=None):
    exporters = []
    if json_path:
        exporters.append(JSONExporter(json_path))
    if tb_path:
        exporters.append(TensorBoardExporter(tb_path))
    return exporters


def cmd_record(args):
    collectors = _build_collectors(include_gpu=not args.cpu_only)
    exporters = _build_exporters(args.json, args.tensorboard)

    with ProfilerManager(collectors, exporters, interval=args.interval):
        if args.duration is None:
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
        else:
            time.sleep(args.duration)


def cmd_launch(args):
    collectors = _build_collectors(include_gpu=not args.cpu_only)
    exporters = _build_exporters(args.json, args.tensorboard)

    if not args.command:
        raise SystemExit("No command provided after -- (e.g.,: ml-profiler launch -- python train.py)")

    with ProfilerManager(collectors, exporters, interval=args.interval):
        proc = subprocess.Popen(args.command)
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait()


def main():
    p = argparse.ArgumentParser(prog="ml-profiler", description="Record CPU/GPU metrics and export them.")
    sub = p.add_subparsers(dest="subcmd", required=True)

    # record
    r = sub.add_parser("record", help="Record metrics for a duration (or until Ctrl+C)")
    r.add_argument("--interval", type=float, default=0.5, help="Sampling interval in seconds")
    r.add_argument("--duration", type=float, default=None, help="Duration to run (seconds). Default: until Ctrl+C")
    r.add_argument("--json", type=str, default=None, help="Path to write JSON results (e.g., logs/profile.json)")
    r.add_argument("--tensorboard", type=str, default=None, help="TensorBoard logdir (e.g., runs/prof)")
    r.add_argument("--cpu-only", action="store_true", help="Skip GPU metrics")
    r.set_defaults(func=cmd_record)

    # launch
    l = sub.add_parser("launch", help="Launch a command and record while it runs")
    l.add_argument("--interval", type=float, default=0.5)
    l.add_argument("--json", type=str, default=None)
    l.add_argument("--tensorboard", type=str, default=None)
    l.add_argument("--cpu-only", action="store_true")
    l.add_argument("--", dest="--", action="store_true")  # visual separator
    l.add_argument("command", nargs=argparse.REMAINDER, help="Command to run after --")
    l.set_defaults(func=cmd_launch)

    args = p.parse_args()
    # normalize command (strip potential leading --)
    if args.subcmd == "launch" and args.command and args.command[0] == "--":
        args.command = args.command[1:]

    Path("logs").mkdir(exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    main()
