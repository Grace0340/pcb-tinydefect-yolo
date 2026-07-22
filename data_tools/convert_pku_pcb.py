"""Convert PKU-Market-PCB (VOC XML) to YOLO format with validity checks.

Expected source layout:
  <src>/images/<DefectDir>/*.jpg
  <src>/Annotations/<DefectDir>/*.xml
Output:
  <dst>/images/{train,val,test}/*.jpg
  <dst>/labels/{train,val,test}/*.txt
  <dst>/data.yaml
  <dst>/conversion_report.json  (counts + discarded samples with reasons)
"""

import argparse
import json
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

CLASSES = ["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"]


def parse_xml(xml_path):
    """Return (width, height, boxes) or raise ValueError with a reason."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    if size is None:
        raise ValueError("missing <size>")
    w = int(size.findtext("width", "0"))
    h = int(size.findtext("height", "0"))
    if w <= 0 or h <= 0:
        raise ValueError(f"invalid image size {w}x{h}")

    boxes = []
    for obj in root.iter("object"):
        name = obj.findtext("name", "").strip().lower()
        if name not in CLASSES:
            raise ValueError(f"unknown class '{name}'")
        bb = obj.find("bndbox")
        xmin = float(bb.findtext("xmin"))
        ymin = float(bb.findtext("ymin"))
        xmax = float(bb.findtext("xmax"))
        ymax = float(bb.findtext("ymax"))
        if not (0 <= xmin < xmax <= w and 0 <= ymin < ymax <= h):
            raise ValueError(f"box out of bounds ({xmin},{ymin},{xmax},{ymax}) in {w}x{h}")
        boxes.append((CLASSES.index(name), xmin, ymin, xmax, ymax))
    if not boxes:
        raise ValueError("no valid objects")
    return w, h, boxes


def to_yolo_lines(w, h, boxes):
    lines = []
    for cls, xmin, ymin, xmax, ymax in boxes:
        cx = (xmin + xmax) / 2 / w
        cy = (ymin + ymax) / 2 / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="PCB_DATASET root")
    ap.add_argument("--dst", required=True, help="output root")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", type=float, nargs=3, default=[0.8, 0.1, 0.1],
                    metavar=("TRAIN", "VAL", "TEST"))
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    img_root = src / "images"
    ann_root = src / "Annotations"
    assert img_root.is_dir() and ann_root.is_dir(), "src must contain images/ and Annotations/"

    valid, discarded = [], []
    for xml_path in sorted(ann_root.rglob("*.xml")):
        stem = xml_path.stem
        candidates = list(img_root.rglob(stem + ".jpg")) + list(img_root.rglob(stem + ".JPG"))
        if not candidates:
            discarded.append({"sample": stem, "reason": "image file not found"})
            continue
        try:
            w, h, boxes = parse_xml(xml_path)
        except (ValueError, ET.ParseError) as e:
            discarded.append({"sample": stem, "reason": str(e)})
            continue
        valid.append((candidates[0], to_yolo_lines(w, h, boxes)))

    rng = random.Random(args.seed)
    rng.shuffle(valid)
    n = len(valid)
    n_train = round(n * args.split[0])
    n_val = round(n * args.split[1])
    splits = {
        "train": valid[:n_train],
        "val": valid[n_train:n_train + n_val],
        "test": valid[n_train + n_val:],
    }

    for split, items in splits.items():
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img_path, lines in items:
            shutil.copy2(img_path, dst / "images" / split / img_path.name)
            (dst / "labels" / split / (img_path.stem + ".txt")).write_text(
                "\n".join(lines) + "\n", encoding="utf-8")

    (dst / "data.yaml").write_text(
        f"path: {dst.resolve().as_posix()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        f"nc: {len(CLASSES)}\nnames: {CLASSES}\n", encoding="utf-8")

    report = {
        "dataset": "PKU-Market-PCB",
        "seed": args.seed,
        "total_annotations": len(valid) + len(discarded),
        "valid": len(valid),
        "discarded_count": len(discarded),
        "discarded": discarded,
        "split_sizes": {k: len(v) for k, v in splits.items()},
    }
    (dst / "conversion_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "discarded"}, indent=2))
    if discarded:
        print(f"NOTE: {len(discarded)} samples discarded, see conversion_report.json")


if __name__ == "__main__":
    main()
