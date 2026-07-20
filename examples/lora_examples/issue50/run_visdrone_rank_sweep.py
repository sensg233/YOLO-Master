"""Run and summarize YOLO-Master-EsMoE-N LoRA rank sweeps on VisDrone.

The script runs r=4,8,16 with alpha=2*r by default, captures each training log,
and writes both CSV and Markdown summaries with mAP50-95, trainable parameter
count, wall-clock training time, and peak GPU memory when those values are
available from Ultralytics outputs.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[3]
CONFIG = Path("examples/lora_examples/issue50/yolo_master_visdrone_lora.yaml")
DEFAULT_RANKS = (4, 8, 16)
DEFAULT_WEIGHTS = Path("examples/lora_examples/issue50/YOLO-Master-EsMoE-N.pt")
DEFAULT_WEIGHTS_URL = (
    "https://github.com/Tencent/YOLO-Master/releases/download/"
    "YOLO-Master-v26.02/YOLO-Master-EsMoE-N.pt"
)


MAP_KEYS = (
    "metrics/mAP50-95(B)",
    "metrics/mAP50-95",
    "mAP50-95(B)",
    "mAP50_95_B",
    "map50_95",
)
MAP50_KEYS = (
    "metrics/mAP50(B)",
    "metrics/mAP50",
    "mAP50(B)",
    "mAP50_B",
    "map50",
)
GPU_KEYS = ("train/GPU_mem", "GPU_mem", "gpu_mem")


def resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def ensure_weights(weights: str | Path, url: str) -> Path:
    path = resolve_repo_path(weights)
    if path.exists():
        print(f"[weights] found {path}")
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".download")
    print(f"[weights] downloading {url}")
    print(f"[weights] target {path}")
    try:
        urllib.request.urlretrieve(url, temp_path)
        temp_path.replace(path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise
    return path


def should_check_model(model: str | None) -> bool:
    if not model:
        return False
    return Path(model).suffix.lower() == ".pt" and "://" not in model


def weights_to_check(args: argparse.Namespace) -> str | Path:
    if args.weights:
        return args.weights
    if should_check_model(args.model):
        return args.model
    return DEFAULT_WEIGHTS


def parse_ranks(raw: str) -> list[int]:
    ranks = [int(item.strip()) for item in raw.split(",") if item.strip()]
    invalid = [rank for rank in ranks if rank <= 0]
    if invalid:
        raise argparse.ArgumentTypeError(f"ranks must be positive, got {invalid}")
    return ranks


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def first_float(row: dict[str, str], keys: Iterable[str]) -> float | None:
    for key in keys:
        if key in row and str(row[key]).strip():
            match = re.search(r"-?\d+(?:\.\d+)?", str(row[key]))
            if match:
                return float(match.group(0))
    return None


def best_metric(rows: list[dict[str, str]], keys: Iterable[str]) -> tuple[float | None, int | None]:
    best_value: float | None = None
    best_epoch: int | None = None
    for index, row in enumerate(rows, start=1):
        value = first_float(row, keys)
        if value is None:
            continue
        if best_value is None or value > best_value:
            best_value = value
            epoch_value = first_float(row, ("epoch",))
            best_epoch = int(epoch_value) if epoch_value is not None else index
    return best_value, best_epoch


def max_metric(rows: list[dict[str, str]], keys: Iterable[str]) -> float | None:
    values = [first_float(row, keys) for row in rows]
    numeric = [value for value in values if value is not None]
    return max(numeric) if numeric else None


def parse_log_stats(log_path: Path) -> dict[str, float | int | None]:
    text = log_path.read_text(encoding="utf-8", errors="ignore") if log_path.exists() else ""

    trainable = None
    match = re.search(r"Trainable:\s*([\d,]+)", text)
    if match:
        trainable = int(match.group(1).replace(",", ""))

    adapter = None
    match = re.search(r"Adapter:\s*([\d,]+)", text)
    if match:
        adapter = int(match.group(1).replace(",", ""))

    peak_gpu = None
    gpu_values = []
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*G", text):
        gpu_values.append(float(match.group(1)))
    if gpu_values:
        peak_gpu = max(gpu_values)

    return {
        "trainable_params": trainable,
        "adapter_params": adapter,
        "peak_gpu_mem_gb_from_log": peak_gpu,
    }


def summarize_run(rank: int, run_dir: Path, log_path: Path, elapsed_s: float, returncode: int) -> dict[str, str]:
    rows = read_csv_rows(run_dir / "results.csv")
    best_map, best_epoch = best_metric(rows, MAP_KEYS)
    best_map50, _ = best_metric(rows, MAP50_KEYS)
    peak_gpu = max_metric(rows, GPU_KEYS)
    log_stats = parse_log_stats(log_path)
    if peak_gpu is None:
        peak_gpu = log_stats["peak_gpu_mem_gb_from_log"]

    return {
        "scene": "visdrone",
        "lora_r": str(rank),
        "lora_alpha": str(rank * 2),
        "epochs": "",
        "mAP50": "" if best_map50 is None else f"{best_map50:.5f}",
        "mAP50-95": "" if best_map is None else f"{best_map:.5f}",
        "best_epoch": "" if best_epoch is None else str(best_epoch),
        "trainable_params": "" if log_stats["trainable_params"] is None else str(log_stats["trainable_params"]),
        "adapter_params": "" if log_stats["adapter_params"] is None else str(log_stats["adapter_params"]),
        "train_time_min": f"{elapsed_s / 60:.2f}",
        "peak_gpu_mem_gb": "" if peak_gpu is None else f"{float(peak_gpu):.3f}",
        "returncode": str(returncode),
        "run_dir": str(run_dir).replace("\\", "/"),
        "log": str(log_path).replace("\\", "/"),
    }


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scene",
        "lora_r",
        "lora_alpha",
        "epochs",
        "mAP50",
        "mAP50-95",
        "best_epoch",
        "trainable_params",
        "adapter_params",
        "train_time_min",
        "peak_gpu_mem_gb",
        "returncode",
        "run_dir",
        "log",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# VisDrone LoRA Rank Sweep Results",
        "",
        "| Rank | Alpha | mAP50 | mAP50-95 | Best epoch | Trainable params | Adapter params | Train time (min) | Peak GPU (GB) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {lora_r} | {lora_alpha} | {mAP50} | {mAP50-95} | {best_epoch} | "
            "{trainable_params} | {adapter_params} | {train_time_min} | {peak_gpu_mem_gb} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- Empty metric cells mean the training command did not produce a parseable `results.csv` or log value.",
            "- `train_time_min` is measured by this wrapper, so it includes process startup and final validation time.",
            "- Routing/gating modules are excluded by `yolo_master_visdrone_lora.yaml`; only expert/visual transform paths remain eligible.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_override(command: list[str], key: str, value: object | None) -> None:
    if value is not None:
        command.append(f"{key}={value}")


def build_command(args: argparse.Namespace, rank: int, name: str) -> list[str]:
    command = [
        args.yolo_bin,
        "train",
        f"cfg={CONFIG.as_posix()}",
        f"lora_r={rank}",
        f"lora_alpha={rank * 2}",
        f"name={name}",
    ]
    append_override(command, "project", args.project)
    append_override(command, "epochs", args.epochs)
    append_override(command, "batch", args.batch)
    append_override(command, "imgsz", args.imgsz)
    append_override(command, "device", args.device)
    append_override(command, "model", args.model)
    if args.exist_ok:
        command.append("exist_ok=True")
    return command


def run_command_with_tee(command: list[str], log_path: Path) -> int:
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        if process.stdout is not None:
            for line in process.stdout:
                print(line, end="")
                log.write(line)
                log.flush()
        return process.wait()


def run_one(args: argparse.Namespace, rank: int) -> dict[str, str]:
    name = f"yolo_master_visdrone_lora_r{rank}"
    project = args.project or "runs/lora_examples/issue50"
    run_dir = ROOT / project / name
    log_dir = ROOT / args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    command = build_command(args, rank, name)

    if args.dry_run:
        print(" ".join(command))
        return {
            "scene": "visdrone",
            "lora_r": str(rank),
            "lora_alpha": str(rank * 2),
            "epochs": "" if args.epochs is None else str(args.epochs),
            "mAP50": "",
            "mAP50-95": "",
            "best_epoch": "",
            "trainable_params": "",
            "adapter_params": "",
            "train_time_min": "",
            "peak_gpu_mem_gb": "",
            "returncode": "dry-run",
            "run_dir": str(run_dir).replace("\\", "/"),
            "log": str(log_path).replace("\\", "/"),
        }

    start = time.perf_counter()
    returncode = run_command_with_tee(command, log_path)
    elapsed_s = time.perf_counter() - start
    summary = summarize_run(rank, run_dir, log_path, elapsed_s, returncode)
    if args.epochs is not None:
        summary["epochs"] = str(args.epochs)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranks", type=parse_ranks, default=list(DEFAULT_RANKS), help="Comma-separated ranks, default: 4,8,16")
    parser.add_argument("--epochs", type=int, default=None, help="Override YAML epochs; must be 20-50 when provided")
    parser.add_argument("--batch", type=int, default=None, help="Override YAML batch")
    parser.add_argument("--imgsz", type=int, default=None, help="Override YAML imgsz")
    parser.add_argument("--device", default=None, help="Override YAML device")
    parser.add_argument("--project", default=None, help="Override YAML project")
    parser.add_argument("--model", default=None, help="Override YAML model/weights path")
    parser.add_argument("--weights", default=None, help="Weights file to check/download")
    parser.add_argument("--weights-url", default=DEFAULT_WEIGHTS_URL, help="URL used when --weights is missing")
    parser.add_argument("--skip-weights-check", action="store_true", help="Do not check/download pretrained weights")
    parser.add_argument("--log-dir", default="examples/lora_examples/issue50/logs")
    parser.add_argument("--output", default="examples/lora_examples/issue50/visdrone_rank_sweep_results.csv")
    parser.add_argument("--markdown", default="examples/lora_examples/issue50/visdrone_rank_sweep_results.md")
    parser.add_argument("--yolo-bin", default="yolo")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.epochs is not None and not 20 <= args.epochs <= 50:
        parser.error("--epochs must be between 20 and 50 for this quick-iteration benchmark")

    if not args.dry_run and not args.skip_weights_check:
        ensure_weights(weights_to_check(args), args.weights_url)

    rows = [run_one(args, rank) for rank in args.ranks]
    write_csv(ROOT / args.output, rows)
    write_markdown(ROOT / args.markdown, rows)
    failed = [row for row in rows if row["returncode"] not in {"0", "dry-run"}]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
