#!/usr/bin/env python3
"""
YOLOv8 training script for the prepared dataset.

- Installs Ultralytics if missing
- Trains YOLOv8 (default: yolov8m.pt) on dataset defined by data.yaml
- Uses MPS (Apple Metal) device by default
- Batch-size fallback if memory issues occur
- Prints best weights path and final metrics (mAP50-95, precision, recall)

Usage examples:
  python3.11 train_yolo.py
  python3.11 train_yolo.py --model yolov8n.pt --epochs 100 --batch 8 --imgsz 640
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def ensure_ultralytics_installed() -> None:
    try:
        import ultralytics  # noqa: F401
    except Exception:
        print("Ultralytics not found. Installing ultralytics...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])  # non-interactive
        except subprocess.CalledProcessError as e:
            # Handle Homebrew/PEP 668 externally-managed-environment by retrying with user + break flags
            print("pip install failed; retrying with --user and --break-system-packages...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--user", "--break-system-packages", "ultralytics"
            ])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8 on prepared dataset")
    parser.add_argument("--model", type=str, default="yolov8m.pt", help="Model weights to start from")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Training batch size (fallback if OOM)")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument("--data", type=str, default="dataset/data.yaml", help="Path to data.yaml")
    parser.add_argument("--device", type=str, default="mps", help="Compute device: 'mps', 'cpu', or CUDA index")
    return parser.parse_args()


def train_with_fallback(model, data_yaml: str, epochs: int, imgsz: int, batch: int, device: str):
    """Attempt training with the requested batch size, and fall back if memory errors occur.

    Strategy:
    - If device is mps, set PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 to relax MPS cap.
    - Try initial (batch, imgsz)
    - On OOM: progressively reduce batch down to 1
    - If still OOM: reduce imgsz stepwise [640, 512, 416] and retry batches again
    """
    from ultralytics import YOLO

    if str(device).lower() == "mps":
        os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

    def _run_train(bs: int, size: int):
        print(f"Starting training with batch={bs}, epochs={epochs}, imgsz={size}, device={device}")
        return model.train(data=data_yaml, epochs=epochs, imgsz=size, batch=bs, device=device)

    def _is_oom(err: Exception) -> bool:
        m = str(err).lower()
        return ("out of memory" in m) or ("oom" in m) or ("mps" in m and "alloc" in m)

    sizes = [imgsz] if imgsz not in (640, 512, 416) else [imgsz] + [s for s in [640, 512, 416] if s != imgsz]
    # Ensure unique order preference 640->512->416
    if sizes[0] != 640:
        sizes = [640, 512, 416]

    for size in sizes:
        bs = batch
        while bs >= 1:
            try:
                return _run_train(bs, size)
            except RuntimeError as e:
                print(f"Training failed with batch={bs}, imgsz={size}: {e}")
                if _is_oom(e):
                    new_bs = bs // 2 if bs > 1 else 0
                    if new_bs >= 1:
                        print(f"Retrying with smaller batch size: batch={new_bs}")
                        bs = new_bs
                        continue
                    else:
                        print("Cannot reduce batch further; will try smaller image size if available.")
                        break
                raise
        print("Trying next smaller image size...")
    raise RuntimeError("Exhausted all fallback options (batch and imgsz) due to OOM.")


def main() -> None:
    ensure_ultralytics_installed()
    from ultralytics import YOLO

    args = parse_args()

    # Load model
    model = YOLO(args.model)

    # Device: Apple Metal GPU (MPS)
    device = args.device

    # Train with batch-size fallback handling
    results = train_with_fallback(
        model=model,
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
    )

    # Print training summary
    save_dir = getattr(getattr(model, "trainer", None), "save_dir", None)
    if save_dir is None:
        # Fallback: use Ultralytics default runs directory if trainer missing
        save_dir = Path("runs/detect/train")
    else:
        save_dir = Path(save_dir)

    best_weights = save_dir / "weights" / "best.pt"
    print("\nTraining completed.")
    print(f"Run directory: {save_dir}")
    print(f"Best weights: {best_weights}")

    # Validate to get final metrics on the validation set
    print("\nRunning validation to compute final metrics on the val split...")
    metrics = model.val(data=args.data, imgsz=args.imgsz, device=device, split="val")

    # Ultralytics returns a metrics dict-like object with keys such as:
    # 'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)'
    # We'll attempt to fetch these robustly.
    def _get_metric(m, keys):
        for k in keys:
            if k in m:
                return m[k]
        return None

    try:
        metrics_dict = dict(metrics)
    except Exception:
        # Some versions may return an object with .results_dict
        metrics_dict = getattr(metrics, "results_dict", {}) or {}

    precision = _get_metric(metrics_dict, ["metrics/precision(B)", "precision"])
    recall = _get_metric(metrics_dict, ["metrics/recall(B)", "recall"])
    map_50_95 = _get_metric(metrics_dict, ["metrics/mAP50-95(B)", "mAP50-95"])  # primary metric

    print("\nFinal metrics (val):")
    print(f"  mAP50-95: {map_50_95}")
    print(f"  precision: {precision}")
    print(f"  recall: {recall}")


if __name__ == "__main__":
    main()


