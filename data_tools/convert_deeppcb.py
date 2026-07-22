"""Convert DeepPCB to YOLO format using the official trainval/test split.

Source layout (https://github.com/tangsanli5201/DeepPCB):
  <src>/PCBData/<groupXXXXX>/<XXXXX>/*_test.jpg       tested images
  <src>/PCBData/<groupXXXXX>/<XXXXX>_not/*.txt        annotations: x1 y1 x2 y2 class(1-6)
  <src>/PCBData/trainval.txt, test.txt                official split lists
Output: YOLO dataset at <dst> + conversion_report.json
"""

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image

CLASSES = ["open", "short", "mousebite", "spur", "copper", "pin-hole"]


def convert_entry(line, pcb_root):
    """One split-list line -> (img_path, yolo_lines) or raise ValueError."""
    parts = line.split()
    img_rel, ann_rel = parts[0], parts[1]
    img_path = pcb_root / img_rel
    if not img_path.exists():
        # split lists reference the template name; tested image carries _test suffix
        alt = img_path.with_name(img_path.stem + "_test" + img_path.suffix)
        if alt.exists():
            img_path = alt
        else:
            raise ValueError(f"image not found: {img_rel}")
    ann_path = pcb_root / ann_rel
    if not ann_path.exists():
        raise ValueError(f"annotation not found: {ann_rel}")

    with Image.open(img_path) as im:
        w, h = im.size

    lines = []
    for raw in ann_path.read_text().strip().splitlines():
        x1, y1, x2, y2, cls = (float(v) for v in raw.split()[:5])
        cls = int(cls) - 1  # 1-based -> 0-based
        if not (0 <= cls < len(CLASSES)):
            raise ValueError(f"invalid class {cls + 1} in {ann_rel}")
        if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
            raise ValueError(f"box out of bounds in {ann_rel}: {raw}")
        lines.append(f"{cls} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                     f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    if not lines:
        raise ValueError(f"no boxes in {ann_rel}")
    return img_path, lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="DeepPCB repo root")
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    pcb_root = src / "PCBData"
    assert pcb_root.is_dir(), "src must contain PCBData/"

    report = {"dataset": "DeepPCB", "splits": {}, "discarded": []}
    split_map = {"trainval.txt": "train", "test.txt": "test"}
    for list_name, split in split_map.items():
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)
        ok = 0
        for line in (pcb_root / list_name).read_text().strip().splitlines():
            try:
                img_path, lines = convert_entry(line, pcb_root)
            except (ValueError, OSError) as e:
                report["discarded"].append({"entry": line, "reason": str(e)})
                continue
            # group dir name disambiguates identical stems across groups
            out_name = f"{img_path.parent.parent.name}_{img_path.name}"
            shutil.copy2(img_path, dst / "images" / split / out_name)
            (dst / "labels" / split / (Path(out_name).stem + ".txt")).write_text(
                "\n".join(lines) + "\n", encoding="utf-8")
            ok += 1
        report["splits"][split] = ok

    (dst / "data.yaml").write_text(
        f"path: {dst.resolve().as_posix()}\n"
        "train: images/train\nval: images/test\ntest: images/test\n"
        f"nc: {len(CLASSES)}\nnames: {CLASSES}\n", encoding="utf-8")

    report["discarded_count"] = len(report["discarded"])
    (dst / "conversion_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "discarded"}, indent=2))


if __name__ == "__main__":
    main()
