"""Step 02: find the cows in every sampled frame (detection only, no tracking).

YOLO26n (Ultralytics ``yolo26n.pt``, trained on the COCO dataset; only class
19 "cow" is kept) draws a box around every cow it sees in a frame, with a
confidence from 0 to 1. Frames are sampled at ``TargetFps`` (2 fps: the
camera only updates the picture ~2 times a second), at ``ImageSize`` 1024:
at YOLO's usual 640 it found the lying cow in only 15 % of frames on
1080-1320 s, at 1024 in 86 %. Boxes are not linked
across frames here; that is step 03 (SAM 2.1).

Step 03 uses these boxes to (1) start SAM on the cow at the beginning of each
chunk and (2) check SAM frame by frame (does YOLO see her where SAM does?).

Outputs in ``OutputDir``, overwritten on every run:

- ``Detections.csv``: one row per box (Frame, TimeSeconds, X1..Y2, Confidence, ClassId)
- ``DetectionSummary.csv``: per minute of the clip, how many sampled frames had a cow
  and how sure YOLO was (shows when she is easy or hard to see)
- ``Detect.json``: the settings (step 03 checks the frame rate matches)
- ``PreviewDetections.mp4`` (``Preview=True``): boxes drawn, sped up ``PreviewSpeed`` times
"""

import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .VideoTools import FormatTime, IterSampledFrames, ProbeVideo

CocoCow = 19
DetectionColumns = ["Frame", "TimeSeconds", "X1", "Y1", "X2", "Y2", "Confidence", "ClassId"]


def Color(Id: int) -> tuple:
    """A fixed colour per track / box number, the same in every preview."""
    Rng = np.random.default_rng(int(Id))
    return tuple(int(C) for C in Rng.integers(64, 256, 3))


def DrawBoxes(Frame, Boxes, Scale, Label):
    """``Boxes``: (Id, X1, Y1, X2, Y2, Text)."""
    Small = cv2.resize(Frame, None, fx=Scale, fy=Scale)
    for Id, X1, Y1, X2, Y2, Text in Boxes:
        P1, P2 = (int(X1 * Scale), int(Y1 * Scale)), (int(X2 * Scale), int(Y2 * Scale))
        cv2.rectangle(Small, P1, P2, Color(Id), 2)
        cv2.putText(Small, Text, (P1[0], max(P1[1] - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, Color(Id), 2)
    cv2.putText(Small, Label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return Small


def RemoveDuplicateBoxes(Boxes: pd.DataFrame, Containment=0.8) -> pd.DataFrame:
    """In each frame, drop a box that lies mostly (``Containment``) inside a
    more confident one (e.g. a second box on the cow's head)."""
    Keep = np.ones(len(Boxes), bool)
    Positions = {Index: Pos for Pos, Index in enumerate(Boxes.index)}
    for _, Frame in Boxes.groupby("Frame"):
        if len(Frame) < 2:
            continue
        Rows = Frame.sort_values("Confidence", ascending=False)
        B = Rows[["X1", "Y1", "X2", "Y2"]].to_numpy()
        Areas = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
        for I in range(1, len(Rows)):
            for J in range(I):
                if not Keep[Positions[Rows.index[J]]]:
                    continue
                W = min(B[I, 2], B[J, 2]) - max(B[I, 0], B[J, 0])
                H = min(B[I, 3], B[J, 3]) - max(B[I, 1], B[J, 1])
                if W > 0 and H > 0 and W * H / max(Areas[I], 1) >= Containment:
                    Keep[Positions[Rows.index[I]]] = False
                    break
    return Boxes[Keep]


def DetectCows(ClipPath, OutputDir, StartSeconds=0, EndSeconds=None, TargetFps=2, ModelName="yolo26n.pt",
               ClassIds=(CocoCow,), Confidence=0.25, ImageSize=1024, Preview=False, PreviewSpeed=4,
               PreviewWidth=960) -> dict:
    """Run YOLO on ``StartSeconds``..``EndSeconds`` of the clip (None = to the end)."""
    from ultralytics import YOLO

    OutputDir = Path(OutputDir)
    OutputDir.mkdir(parents=True, exist_ok=True)
    Info = ProbeVideo(ClipPath)
    EndSeconds = min(EndSeconds or Info["DurationSeconds"], Info["DurationSeconds"])
    print(f"{Info['Video']}: {FormatTime(StartSeconds)}-{FormatTime(EndSeconds)} of the clip at {TargetFps} fps")

    Model = YOLO(ModelName)
    Scale = PreviewWidth / Info["Width"]
    PreviewTemp = OutputDir / "PreviewDetections.tmp.avi"
    PreviewWriter, Rows, Times = None, [], []
    Started = time.time()
    for TimeSeconds, Frame in IterSampledFrames(ClipPath, StartSeconds, EndSeconds - StartSeconds, TargetFps,
                                                Info["Width"], Info["Height"]):
        R = Model.predict(Frame, classes=list(ClassIds) if ClassIds else None, conf=Confidence,
                          imgsz=ImageSize, verbose=False)[0]
        FrameIndex = int(round(TimeSeconds * Info["Fps"]))
        Times.append(TimeSeconds)
        Drawn = []
        if R.boxes is not None and len(R.boxes):
            for I, ((X1, Y1, X2, Y2), Conf, ClassId) in enumerate(zip(
                    R.boxes.xyxy.cpu().numpy().round(1), R.boxes.conf.tolist(), R.boxes.cls.int().tolist())):
                Rows.append([FrameIndex, TimeSeconds, float(X1), float(Y1), float(X2), float(Y2), round(Conf, 3), ClassId])
                Drawn.append((I + 1, X1, Y1, X2, Y2, f"{Conf:.2f}"))
        if Preview:
            Image = DrawBoxes(Frame, Drawn, Scale, f"{TimeSeconds:7.1f}s")
            if PreviewWriter is None:
                PreviewWriter = cv2.VideoWriter(str(PreviewTemp), cv2.VideoWriter_fourcc(*"MJPG"),
                                                TargetFps * PreviewSpeed, (Image.shape[1], Image.shape[0]))
            PreviewWriter.write(Image)
        if len(Times) % 500 == 0:
            print(f"  {TimeSeconds:.0f} s done, {len(Times) / (time.time() - Started):.1f} frames/s")
    del Model

    (OutputDir / "PreviewDetections.mp4").unlink(missing_ok=True)
    if PreviewWriter is not None:
        PreviewWriter.release()
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(PreviewTemp), "-c:v", "libx264", "-preset", "veryfast",
                        "-crf", "28", "-pix_fmt", "yuv420p", str(OutputDir / "PreviewDetections.mp4")], check=True)
        PreviewTemp.unlink()

    Detections = pd.DataFrame(Rows, columns=DetectionColumns)
    Detections.to_csv(OutputDir / "Detections.csv", index=False)
    Settings = {"Clip": str(ClipPath), "StartSeconds": StartSeconds, "EndSeconds": EndSeconds,
                "TargetFps": TargetFps, "ModelName": ModelName, "Confidence": Confidence, "ImageSize": ImageSize}
    (OutputDir / "Detect.json").write_text(json.dumps(Settings, indent=1))

    # Per minute: share of sampled frames with a cow, and how sure YOLO was.
    Minute = lambda T: (np.asarray(T) // 60).astype(int)
    Frames = pd.Series(Minute(Times)).value_counts().sort_index().rename("SampledFrames")
    Best = Detections.groupby("TimeSeconds").Confidence.max().reset_index()
    Best["Minute"] = Minute(Best.TimeSeconds)
    PerMinute = Best.groupby("Minute").agg(FramesWithCow=("TimeSeconds", "count"), MeanBestConfidence=("Confidence", "mean"))
    Summary = pd.concat([Frames, PerMinute], axis=1).fillna({"FramesWithCow": 0}).rename_axis("Minute").reset_index()
    Summary["CowShare"] = (Summary.FramesWithCow / Summary.SampledFrames).round(3)
    Summary["MeanBestConfidence"] = Summary.MeanBestConfidence.round(3)
    Summary.insert(1, "ClipTime", [FormatTime(M * 60) for M in Summary.Minute])
    Summary.to_csv(OutputDir / "DetectionSummary.csv", index=False)

    Seconds = time.time() - Started
    Stats = {"SampledFrames": len(Times), "FramesWithCow": int(Detections.TimeSeconds.nunique()),
             "Boxes": len(Detections), "Seconds": round(Seconds, 1),
             "FramesPerSecond": round(len(Times) / Seconds, 1) if Seconds else 0}
    print(Stats)
    return Stats


def LoadDetections(DetectionsDir) -> tuple:
    """``(Detections, Settings)`` from a step 02 folder."""
    DetectionsDir = Path(DetectionsDir)
    return (pd.read_csv(DetectionsDir / "Detections.csv"),
            json.loads((DetectionsDir / "Detect.json").read_text()))
