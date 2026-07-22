"""Evaluate a trained checkpoint on the held-out test split.

Example:
  python eval.py --weights runs/p2light_nwd_seed0/weights/best.pt \
      --data datasets/pku_pcb/data.yaml
"""

import argparse
import json

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--per-class", action="store_true")
    args = ap.parse_args()

    model = YOLO(args.weights)
    m = model.val(data=args.data, split=args.split, batch=args.batch,
                  imgsz=args.imgsz, verbose=False)
    out = {
        "P": round(float(m.box.mp), 4),
        "R": round(float(m.box.mr), 4),
        "mAP50": round(float(m.box.map50), 4),
        "mAP50_95": round(float(m.box.map), 4),
        "params_M": round(sum(p.numel() for p in model.model.parameters()) / 1e6, 3),
    }
    if args.per_class:
        out["per_class"] = {
            model.names[int(c)]: {
                "AP50": round(float(m.box.ap50[i]), 4),
                "AP50_95": round(float(m.box.ap[i]), 4),
            }
            for i, c in enumerate(m.box.ap_class_index)
        }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
