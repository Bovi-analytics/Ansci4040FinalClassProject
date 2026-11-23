"""Canonical crop of one masked cow (stage 4 of the re-ID design).

``CanonicalCrop`` is ported from ``canonical_crop`` in Tol_03_track_v4.py /
Tol_32_dense_crops.py of Sonam525/Re-id-for-tracking (MIT licence, see
NOTICE.md in this folder), with one fix: the sign of the rotation (see the
comment in the code). Keep it unchanged from now on: the gallery and the test
crops must come from exactly the same definition, or their fingerprints are
not comparable.

What it does: background set to a flat fill colour (124, 116, 104); the body's
long axis turned vertical (PCA of the mask pixels); a 2-98 % contrast stretch
over the cow's own pixels (so day and night/IR crops look alike); cut to the
mask with a 10 px margin and resized to 224 x 224.

``CropStats`` adds what the gates and the non-cow filter need: mask area,
elongation, the share of clearly coloured pixels (staff overalls; inert in
night/IR crops, which are grey), as in Tol_54 (S >= 0.35 and V >= 0.5, the
mask edge eroded by 3 px).
"""

import numpy as np
from PIL import Image

MinMaskPixels = 8000
Fill = (124, 116, 104)


def _Bbox(Mask):
    Ys, Xs = np.where(Mask)
    if len(Xs) == 0:
        return None
    return [int(Xs.min()), int(Ys.min()), int(Xs.max()), int(Ys.max())]


def CanonicalCrop(Pil, Mask, Pad=10, OutSize=224, ReturnMask=False):
    """Pose- and exposure-normalised masked crop (``Pil``: RGB image, ``Mask``:
    bool array of the same size). None if the mask is under ``MinMaskPixels``.
    ``ReturnMask``: also return the cow's pixels in the 224 x 224 crop."""
    if Mask is None or Mask.sum() < MinMaskPixels:
        return (None, None) if ReturnMask else None
    Ys, Xs = np.where(Mask)
    Points = np.stack([Xs, Ys], 1).astype(np.float32)
    Points -= Points.mean(0)
    try:
        W, V = np.linalg.eigh(np.cov(Points.T))
        Axis = V[:, int(np.argmax(W))]
        Angle = float(np.degrees(np.arctan2(Axis[1], Axis[0])))
    except Exception:
        Angle = 0.0
    Array = np.array(Pil)
    Fg = np.zeros(Array.shape[:2], np.uint8)
    Fg[Mask] = 255
    Masked = np.empty_like(Array)
    Masked[:] = np.array(Fill, Array.dtype)
    Masked[Mask] = Array[Mask]
    # Changed from the source (90 - Angle): image rows grow downwards and PIL turns anticlockwise, so
    # 90 - Angle only stands the body up when it is already horizontal or vertical (a cow at 30 degrees
    # stayed tilted, one at 135 came out lying). 90 + Angle stands it up at any angle (tested).
    Rotation = 90.0 + Angle
    Rotated = Image.fromarray(Masked).rotate(Rotation, resample=Image.BILINEAR, expand=True, fillcolor=tuple(Fill))
    FgRotated = Image.fromarray(Fg).rotate(Rotation, resample=Image.NEAREST, expand=True, fillcolor=0)
    FgArray = np.array(FgRotated) > 127
    Box = _Bbox(FgArray)
    if Box is None:
        return (None, None) if ReturnMask else None
    A = np.array(Rotated).astype(np.float32)
    Selected = FgArray
    if Selected.sum() > 50:
        Values = A[Selected]
        Low, High = np.percentile(Values, 2), np.percentile(Values, 98)
        if High > Low + 1e-3:
            A[Selected] = np.clip((A[Selected] - Low) * 255.0 / (High - Low), 0, 255)
    A[~Selected] = np.array(Fill, np.float32)
    H, W_ = A.shape[:2]
    Window = (slice(max(0, Box[1] - Pad), min(H, Box[3] + Pad + 1)), slice(max(0, Box[0] - Pad), min(W_, Box[2] + Pad + 1)))
    Crop = Image.fromarray(A[Window].astype(np.uint8)).resize((OutSize, OutSize), Image.BILINEAR)
    if not ReturnMask:
        return Crop
    CropMask = np.array(Image.fromarray(FgArray[Window].astype(np.uint8) * 255).resize((OutSize, OutSize), Image.NEAREST)) > 127
    return Crop, CropMask


def Elongation(Mask) -> float:
    """sqrt(largest / smallest spread) of the mask pixels: ~1 round, 2+ long."""
    Ys, Xs = np.where(Mask)
    if len(Xs) < 10:
        return float("nan")
    W = np.linalg.eigvalsh(np.cov(np.stack([Xs, Ys]).astype(np.float32)))
    return float(np.sqrt(max(W[-1], 1e-6) / max(W[0], 1e-6)))


def ColouredFraction(Crop, CropMask, SatMin=0.35, ValueMin=0.5, Erode=3, MinPixels=500) -> float:
    """Share of the cow's pixels (edge eroded) that are clearly coloured."""
    import cv2

    Fg = cv2.erode(CropMask.astype(np.uint8), np.ones((2 * Erode + 1, 2 * Erode + 1), np.uint8)) > 0
    if Fg.sum() < MinPixels:
        return float("nan")
    Rgb = np.asarray(Crop, np.float32)[Fg] / 255.0
    Max, Min = Rgb.max(1), Rgb.min(1)
    Sat = np.where(Max > 0, (Max - Min) / np.maximum(Max, 1e-6), 0)
    return float(((Sat >= SatMin) & (Max >= ValueMin)).mean())
