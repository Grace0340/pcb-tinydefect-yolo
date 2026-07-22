"""NWD (Normalized Wasserstein Distance) regression loss patch for Ultralytics.

Blends NWD into the box regression similarity inside BboxLoss only, by
replacing the `bbox_iou` symbol in ultralytics.utils.loss's namespace.
The task-aligned assigner (ultralytics.utils.tal) imports its own copy and is
NOT affected, so label assignment stays identical across ablation rows.

blended = alpha * CIoU + (1 - alpha) * NWD;  loss_iou = 1 - blended
alpha = 1.0 -> stock CIoU;  0.5 -> mixed;  0.0 -> pure NWD.

NWD follows Wang et al., "A Normalized Gaussian Wasserstein Distance for
Tiny Object Detection" (arXiv:2110.13389): boxes modeled as 2D Gaussians,
NWD = exp(-sqrt(W2^2) / C) with C = 12.8.
"""

import math

import torch

from ultralytics.utils import loss as uloss
from ultralytics.utils.metrics import bbox_iou as _orig_bbox_iou

_ALPHA = 1.0
_C = 12.8


def set_alpha(alpha):
    global _ALPHA
    _ALPHA = float(alpha)


def _nwd(box1, box2, xywh=True, eps=1e-7):
    if xywh:
        cx1, cy1, w1, h1 = box1.chunk(4, -1)
        cx2, cy2, w2, h2 = box2.chunk(4, -1)
    else:
        x11, y11, x12, y12 = box1.chunk(4, -1)
        x21, y21, x22, y22 = box2.chunk(4, -1)
        cx1, cy1, w1, h1 = (x11 + x12) / 2, (y11 + y12) / 2, x12 - x11, y12 - y11
        cx2, cy2, w2, h2 = (x21 + x22) / 2, (y21 + y22) / 2, x22 - x21, y22 - y21
    w2_dist = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2 + ((w1 - w2) ** 2 + (h1 - h2) ** 2) / 4
    return torch.exp(-torch.sqrt(w2_dist.clamp(min=eps)) / _C)


def _blended_bbox_iou(box1, box2, xywh=True, **kwargs):
    base = _orig_bbox_iou(box1, box2, xywh=xywh, **kwargs)
    if _ALPHA >= 1.0:
        return base
    nwd = _nwd(box1, box2, xywh=xywh)
    if nwd.shape != base.shape:
        nwd = nwd.reshape(base.shape)
    return _ALPHA * base + (1.0 - _ALPHA) * nwd


def apply_patch():
    uloss.bbox_iou = _blended_bbox_iou
    print(f"NWD_PATCH_APPLIED alpha={_ALPHA} C={_C}", flush=True)
