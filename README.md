# pcb-tinydefect-yolo

Reproducible training and evaluation pipeline for tiny-defect detection on
printed circuit boards with YOLOv11-family models. Includes P2-head /
P5-removal architecture variants, an optional Normalized Wasserstein
Distance (NWD) box regression loss, a full factorial experiment grid, and
speed benchmarks.

## Setup

```bash
pip install -r requirements.txt
```

Tested with Python 3.10+, PyTorch 2.3.0, Ultralytics 8.4.98, on a single
NVIDIA RTX 4090D.

## Datasets

Both datasets are public. Download, then convert to YOLO format:

**PKU-Market-PCB** (https://robotics.pku.edu.cn/openlab/datasets/) — use the
693 original annotated images (the rotation-augmented copies in the archive
have no annotation files and are excluded by the converter automatically,
since only samples with a paired XML are kept):

```bash
python data_tools/convert_pku_pcb.py --src /path/to/PCB_DATASET --dst datasets/pku_pcb --seed 0
```

**DeepPCB** (https://github.com/tangsanli5201/DeepPCB) — official
trainval/test split:

```bash
git clone --depth 1 https://github.com/tangsanli5201/DeepPCB
python data_tools/convert_deeppcb.py --src DeepPCB --dst datasets/deeppcb
```

Both converters run validity checks (paired annotation, in-bounds boxes,
known classes) and write a `conversion_report.json` listing anything
discarded.

## Model configurations

| Config | Detection heads |
|---|---|
| `yolo11n.pt` (stock) | P3, P4, P5 |
| `configs/yolo11n-p2.yaml` | P2, P3, P4, P5 |
| `configs/yolo11n-nop5.yaml` | P3, P4 |
| `configs/yolo11n-p2light.yaml` | P2, P3, P4 |

Custom configurations are initialized by transferring all shape-compatible
layers from the official `yolo11n.pt` checkpoint (`--weights yolo11n.pt`).

`patch_nwd.py` swaps the box-similarity function inside the regression loss
for a CIoU/NWD blend controlled by `--alpha` (1.0 = stock CIoU, 0.0 = pure
NWD). The task-aligned assigner is untouched, so label assignment is
identical across all configurations.

## Training

Single run:

```bash
python train.py --model configs/yolo11n-p2light.yaml --weights yolo11n.pt \
    --data datasets/pku_pcb/data.yaml --name p2light_nwd_seed0 --seed 0 --alpha 0.0
```

Full grid (factorial ablation + multi-seed replication + baselines +
DeepPCB generalization; skips runs whose `best.pt` already exists and writes
`results_summary.csv`):

```bash
python run_experiments.py
```

Protocol for every run: 640x640 input, up to 300 epochs with early stopping
(patience 50), batch 16, SGD with the Ultralytics default schedule, default
augmentation, deterministic mode, fixed seed.

## Evaluation

```bash
python eval.py --weights runs/p2light_nwd_seed0/weights/best.pt \
    --data datasets/pku_pcb/data.yaml --per-class
```

## Speed benchmark

```bash
python speed_bench.py --weights runs/p2light_nwd_seed0/weights/best.pt --device cuda
python speed_bench.py --weights runs/p2light_nwd_seed0/weights/best.pt --device cpu
```

## License

MIT
