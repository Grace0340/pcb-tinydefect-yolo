"""Measure parameters, GFLOPs, and batch-1 latency / FPS at 640x640.

GPU protocol: 30 warmup + 200 timed forward passes on a random tensor with
torch.cuda.synchronize around the timed block. Use --device cpu for a
CPU measurement (median of 50 runs after 10 warmup passes).

Example:
  python speed_bench.py --weights runs/p2light_nwd_seed0/weights/best.pt
"""

import argparse
import json
import time

import numpy as np
import torch
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, nargs="+")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    for w in args.weights:
        model = YOLO(w)
        n_params = sum(p.numel() for p in model.model.parameters())
        net = model.model.to(args.device).eval().float()
        x = torch.randn(1, 3, args.imgsz, args.imgsz, device=args.device)

        with torch.inference_mode():
            if args.device == "cuda":
                for _ in range(30):
                    net(x)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(200):
                    net(x)
                torch.cuda.synchronize()
                lat = (time.perf_counter() - t0) / 200 * 1000
            else:
                for _ in range(10):
                    net(x)
                ts = []
                for _ in range(50):
                    t0 = time.perf_counter()
                    net(x)
                    ts.append((time.perf_counter() - t0) * 1000)
                lat = float(np.median(ts))

        try:
            from ultralytics.utils.torch_utils import get_flops
            gflops = round(float(get_flops(net, args.imgsz)), 2)
        except Exception:
            gflops = None

        print(json.dumps({
            "weights": w,
            "device": args.device,
            "params_M": round(n_params / 1e6, 3),
            "GFLOPs": gflops,
            "latency_ms": round(lat, 2),
            "FPS": round(1000 / lat, 1),
        }))


if __name__ == "__main__":
    main()
