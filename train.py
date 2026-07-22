"""Train one configuration.

Examples:
  python train.py --model yolo11n.pt --data datasets/pku_pcb/data.yaml --name base_seed0 --seed 0
  python train.py --model configs/yolo11n-p2light.yaml --weights yolo11n.pt \
      --data datasets/pku_pcb/data.yaml --name p2light_nwd_seed0 --seed 0 --alpha 0.0
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

import patch_nwd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="model yaml under configs/ or a .pt checkpoint")
    ap.add_argument("--weights", default=None,
                    help="optional checkpoint to transfer shape-compatible layers from")
    ap.add_argument("--data", required=True, help="dataset data.yaml")
    ap.add_argument("--name", required=True, help="run name under runs/")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="box loss mix: 1.0 = CIoU, 0.0 = pure NWD, 0.5 = blend")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--runs", default="runs")
    args = ap.parse_args()

    patch_nwd.set_alpha(args.alpha)
    patch_nwd.apply_patch()

    model = YOLO(args.model)
    if args.weights:
        model.load(args.weights)
    model.train(
        data=args.data, name=args.name, imgsz=args.imgsz, epochs=args.epochs,
        patience=args.patience, batch=args.batch, seed=args.seed,
        deterministic=True, project=args.runs, exist_ok=True, workers=8,
    )
    print("TRAIN_DONE", Path(args.runs) / args.name)


if __name__ == "__main__":
    main()
