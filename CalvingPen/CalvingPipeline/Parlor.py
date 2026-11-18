"""Milking-parlor videos: a first look, then the calving steps on a short window.

The parlor camera looks down on the entry: cows walk in fast and stand
shoulder to shoulder in herringbone stalls on both sides. Before building
anything parlor-specific, ``FirstLook`` shows what the detector sees:

- ``FirstLook.json``: the video's resolution, frame rate and length,
- ``FirstLook.jpg``: ``Samples`` frames spread over the video, each with
  YOLO's boxes and the number of cows found,
- ``FirstLook.csv``: per sample frame, cows found and their confidences.

Then the calving steps run unchanged on a short window (PipelineSteps-style
cells in ``ParlorTest.ipynb``): 02 YOLO at a higher frame rate (cows move
fast), 03 SAM 2.1 tracks every cow, 04 masked crops. Outputs go to
``ParlorPipelineOutputs/<video>/``, never into the calving folders.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .Detect import CocoCow, Color
from .VideoTools import FormatTime, GrabFrameAt, ProbeVideo


def FirstLook(VideoPath, OutputDir, Samples=12, ModelName="yolo26m.pt", Confidence=0.25, ImageSize=1024,
              Columns=4, TileWidth=480) -> pd.DataFrame:
    """Sample frames with YOLO boxes; returns one row per sample (Time, Cows, MeanConfidence)."""
    from ultralytics import YOLO

    OutputDir = Path(OutputDir)
    OutputDir.mkdir(parents=True, exist_ok=True)
    Info = ProbeVideo(VideoPath)
    (OutputDir / "FirstLook.json").write_text(json.dumps(Info, indent=1))
    print(f"{Info['Video']}: {Info['Width']}x{Info['Height']}, {Info['Fps']} fps, {Info['Duration']}, {Info['SizeMb']} MB")
    Model = YOLO(ModelName)
    Rows, Tiles = [], []
    for Time in np.linspace(0, max(Info["DurationSeconds"] - 1, 0), Samples):
        try:
            Frame = GrabFrameAt(VideoPath, Time)
        except Exception:
            continue                                   # past the real end (wrong length in the header)
        R = Model.predict(Frame, classes=[CocoCow], conf=Confidence, imgsz=ImageSize, verbose=False)[0]
        Boxes = R.boxes.xyxy.cpu().numpy() if R.boxes is not None and len(R.boxes) else np.zeros((0, 4))
        Conf = R.boxes.conf.cpu().numpy() if len(Boxes) else np.zeros(0)
        for I, (Box, C) in enumerate(zip(Boxes, Conf)):
            X1, Y1, X2, Y2 = map(int, Box)
            cv2.rectangle(Frame, (X1, Y1), (X2, Y2), Color(I + 1), 2)
            cv2.putText(Frame, f"{C:.2f}", (X1, max(Y1 - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, Color(I + 1), 1)
        Label = f"{FormatTime(Time)}  {len(Boxes)} cows"
        cv2.putText(Frame, Label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(Frame, Label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        Scale = TileWidth / Frame.shape[1]
        Tiles.append(cv2.resize(Frame, None, fx=Scale, fy=Scale))
        Rows.append({"TimeSeconds": round(float(Time), 1), "Time": FormatTime(Time), "Cows": len(Boxes),
                     "Confident": int((Conf >= 0.5).sum()), "MeanConfidence": round(float(Conf.mean()), 3) if len(Conf) else 0.0,
                     "MedianBoxWidth": int(np.median(Boxes[:, 2] - Boxes[:, 0])) if len(Boxes) else 0})
    if Tiles:
        Tiles += [np.zeros_like(Tiles[0])] * (-len(Tiles) % Columns)
        Sheet = np.vstack([np.hstack(Tiles[I:I + Columns]) for I in range(0, len(Tiles), Columns)])
        cv2.imwrite(str(OutputDir / "FirstLook.jpg"), Sheet)
    Table = pd.DataFrame(Rows)
    Table.to_csv(OutputDir / "FirstLook.csv", index=False)
    return Table
