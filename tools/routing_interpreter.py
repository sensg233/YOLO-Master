"""Generate routing diagnostics and heatmaps from a YOLO-Master checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="YOLO checkpoint or model YAML")
    parser.add_argument("image", type=Path, help="input image")
    parser.add_argument("--layer", help="exact routed layer name; omit to capture all leaf routed layers")
    parser.add_argument("--expert", type=int, help="also run a forced-expert counterfactual for --layer")
    parser.add_argument("--imgsz", type=int, default=640, help="square inference size")
    parser.add_argument("--device", default="cpu", help="torch device, for example cpu, mps, or cuda:0")
    parser.add_argument("--half", action="store_true", help="use float16 inference (CUDA recommended)")
    parser.add_argument("--output", type=Path, default=Path("runs/routing_interpreter"), help="output directory")
    return parser


def _load_batch(image_path: Path, imgsz: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Load one image using the same letterbox and channel conventions as prediction."""
    from ultralytics.data.augment import LetterBox
    from ultralytics.utils.patches import imread

    image = imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"could not read image: {image_path}")
    resized = LetterBox(new_shape=(imgsz, imgsz), auto=False, stride=32)(image=image)
    rgb = np.ascontiguousarray(resized[..., ::-1].transpose(2, 0, 1))
    return torch.from_numpy(rgb).unsqueeze(0).to(device=device, dtype=dtype).div_(255.0)


def _load_model(model_path: Path, device: torch.device, half: bool) -> torch.nn.Module:
    """Load detection YAMLs and checkpoints without importing unrelated model families."""
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.patches import torch_load

    suffix = model_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        model = DetectionModel(str(model_path), ch=3, verbose=False)
    elif suffix in {".pt", ".pth"}:
        checkpoint = torch_load(model_path, map_location="cpu")
        if isinstance(checkpoint, dict):
            model = checkpoint.get("ema") or checkpoint.get("model")
        else:
            model = checkpoint
        if not isinstance(model, torch.nn.Module):
            raise TypeError(f"checkpoint does not contain an nn.Module under 'ema' or 'model': {model_path}")
    else:
        raise ValueError(f"model must be a .pt, .pth, .yaml, or .yml file, got: {model_path}")
    model = model.to(device).eval()
    for module in model.modules():
        if hasattr(module, "inplace"):
            module.inplace = True
        elif isinstance(module, torch.nn.Upsample) and not hasattr(module, "recompute_scale_factor"):
            module.recompute_scale_factor = None
    return model.half() if half else model.float()


def main(argv: list[str] | None = None) -> int:
    """Run routing capture, collapse checks, rendering, and optional causal analysis."""
    args = build_parser().parse_args(argv)
    if args.expert is not None and not args.layer:
        raise SystemExit("--expert requires --layer because counterfactual routing targets one exact layer")
    if args.imgsz <= 0:
        raise SystemExit("--imgsz must be positive")

    from ultralytics.utils.routing_interpreter import RoutingInterpreter

    device = torch.device(args.device)
    dtype = torch.float16 if args.half else torch.float32
    network = _load_model(args.model, device, args.half)
    batch = _load_batch(args.image, args.imgsz, device, dtype)

    interpreter = RoutingInterpreter(network)
    heatmaps = interpreter.visualize_routing(
        batch,
        layer_name=args.layer,
        output_dir=None,
    )
    visualizations = interpreter.save_routing_visualizations(
        heatmaps,
        args.output,
        input_image=batch,
    )
    summaries = interpreter.collect_layer_summaries(heatmaps=heatmaps)
    collapse = interpreter.detect_routing_collapse(heatmaps=heatmaps)
    causal = (
        interpreter.routing_causal_analysis(batch, args.layer, args.expert).to_dict()
        if args.expert is not None
        else None
    )

    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "routing_report.json"
    payload = {
        "model": str(args.model),
        "image": str(args.image),
        "layer": args.layer,
        "heatmaps": {name: heatmap.to_dict() for name, heatmap in heatmaps.items()},
        "visualizations": {
            name: {artifact: str(path) for artifact, path in artifacts.items()}
            for name, artifacts in visualizations.items()
        },
        "summaries": [summary.to_dict() for summary in summaries],
        "collapse": {name: report.to_dict() for name, report in collapse.items()},
        "causal": causal,
    }
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"Routing report: {report_path}")
    print(f"Heatmaps: {len(heatmaps)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
