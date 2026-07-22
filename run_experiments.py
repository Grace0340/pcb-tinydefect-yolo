"""Run the full experiment grid: 2^3 factorial ablation (A = P2 head,
B = P5 branch removed, C = pure NWD box loss), multi-seed replication,
nano baselines under one protocol, and DeepPCB generalization.

Each run trains (skipped if best.pt already exists), then evaluates on the
test split and appends one row to results_summary.csv.
"""

import csv
import json
from pathlib import Path

from ultralytics import YOLO

import patch_nwd

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
OUT_CSV = HERE / "results_summary.csv"

DATA_PKU = str(HERE / "datasets/pku_pcb/data.yaml")
DATA_DEEP = str(HERE / "datasets/deeppcb/data.yaml")
P2 = str(HERE / "configs/yolo11n-p2.yaml")
P2L = str(HERE / "configs/yolo11n-p2light.yaml")
NOP5 = str(HERE / "configs/yolo11n-nop5.yaml")

# (name, model source, transfer weights, data, seed, alpha)
QUEUE = [
    # nano baselines, one shared protocol (seed 0)
    ("yolov5n_base_seed0", "yolov5n.pt", None, DATA_PKU, 0, 1.0),
    ("yolov8n_base_seed0", "yolov8n.pt", None, DATA_PKU, 0, 1.0),
    ("yolov10n_base_seed0", "yolov10n.pt", None, DATA_PKU, 0, 1.0),
    ("yolo12n_base_seed0", "yolo12n.pt", None, DATA_PKU, 0, 1.0),
    # factorial cells, seed 0 (000, 100, 010, 001, 110, 101, 011, 111)
    ("yolo11n_base_seed0", "yolo11n.pt", None, DATA_PKU, 0, 1.0),
    ("yolo11n_p2_seed0", P2, "yolo11n.pt", DATA_PKU, 0, 1.0),
    ("yolo11n_nop5_seed0", NOP5, "yolo11n.pt", DATA_PKU, 0, 1.0),
    ("base_nwd10_seed0", "yolo11n.pt", None, DATA_PKU, 0, 0.0),
    ("yolo11n_p2light_seed0", P2L, "yolo11n.pt", DATA_PKU, 0, 1.0),
    ("abl_AC_p2_nwd_seed0", P2, "yolo11n.pt", DATA_PKU, 0, 0.0),
    ("abl_BC_nop5_nwd_seed0", NOP5, "yolo11n.pt", DATA_PKU, 0, 0.0),
    ("p2light_nwd_seed0", P2L, "yolo11n.pt", DATA_PKU, 0, 0.0),
    # loss-mix sweep on the unmodified baseline
    ("base_nwd05_seed0", "yolo11n.pt", None, DATA_PKU, 0, 0.5),
    # multi-seed replication of the configurations central to the analysis
    ("yolo11n_base_seed1", "yolo11n.pt", None, DATA_PKU, 1, 1.0),
    ("yolo11n_base_seed2", "yolo11n.pt", None, DATA_PKU, 2, 1.0),
    ("yolo11n_p2_seed1", P2, "yolo11n.pt", DATA_PKU, 1, 1.0),
    ("yolo11n_p2_seed2", P2, "yolo11n.pt", DATA_PKU, 2, 1.0),
    ("yolo11n_nop5_seed1", NOP5, "yolo11n.pt", DATA_PKU, 1, 1.0),
    ("yolo11n_nop5_seed2", NOP5, "yolo11n.pt", DATA_PKU, 2, 1.0),
    ("base_nwd10_seed1", "yolo11n.pt", None, DATA_PKU, 1, 0.0),
    ("base_nwd10_seed2", "yolo11n.pt", None, DATA_PKU, 2, 0.0),
    ("yolo11n_p2light_seed1", P2L, "yolo11n.pt", DATA_PKU, 1, 1.0),
    ("yolo11n_p2light_seed2", P2L, "yolo11n.pt", DATA_PKU, 2, 1.0),
    ("p2light_nwd_seed1", P2L, "yolo11n.pt", DATA_PKU, 1, 0.0),
    ("p2light_nwd_seed2", P2L, "yolo11n.pt", DATA_PKU, 2, 0.0),
    # generalization: retrain baseline and final recipe on DeepPCB
    ("deeppcb_yolo11n_seed0", "yolo11n.pt", None, DATA_DEEP, 0, 1.0),
    ("deeppcb_p2light_nwd_seed0", P2L, "yolo11n.pt", DATA_DEEP, 0, 0.0),
]

patch_nwd.apply_patch()


def evaluate(weights, name, data):
    model = YOLO(weights)
    m = model.val(data=data, split="test", batch=8, imgsz=640,
                  project=str(RUNS), name=name + "_testeval", exist_ok=True,
                  verbose=False)
    return {
        "run": name,
        "precision": round(float(m.box.mp), 4),
        "recall": round(float(m.box.mr), 4),
        "mAP50": round(float(m.box.map50), 4),
        "mAP50_95": round(float(m.box.map), 4),
        "params_M": round(sum(p.numel() for p in model.model.parameters()) / 1e6, 3),
    }


def main():
    RUNS.mkdir(exist_ok=True)
    rows = []
    for name, src, transfer, data, seed, alpha in QUEUE:
        patch_nwd.set_alpha(alpha)
        best = RUNS / name / "weights" / "best.pt"
        if not best.exists():
            model = YOLO(src)
            if transfer:
                model.load(transfer)
            model.train(data=data, name=name, imgsz=640, epochs=300,
                        patience=50, batch=16, seed=seed, deterministic=True,
                        project=str(RUNS), exist_ok=True, workers=8,
                        verbose=False)
        patch_nwd.set_alpha(1.0)  # evaluation always uses stock metrics
        row = evaluate(str(best), name, data)
        row["loss_alpha"] = alpha
        row["seed"] = seed
        rows.append(row)
        print("RUN_DONE", name, json.dumps(row), flush=True)

    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
