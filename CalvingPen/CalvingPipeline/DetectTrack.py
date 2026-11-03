"""Step 02: find cows in a clip and give each one a track ID.

YOLO (COCO "cow" class) boxes the cows in each processed frame, and the
Ultralytics ByteTrack tracker links boxes across frames into tracks. A
TrackId only means "the same cow since she came into view"; it is not an
AnimalId. One cow can get several TrackIds (occluded, left the frame);
Re-ID labels each track later.

Outputs in ``OutputDir``, overwritten on every run:

- ``Tracks.csv``: one row per box per processed frame
- ``TrackSummary.csv``: one row per track (first/last time, frames, confidence)
- ``PreviewTracks.mp4``: the first ``PreviewSeconds`` with boxes and IDs drawn
"""

import csv
import gc
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .VideoTools import ProbeVideo

CocoCow = 19
TrackColumns = ["Frame", "TimeSeconds", "TrackId", "X1", "Y1", "X2", "Y2", "Confidence", "ClassId"]


def EstimateRealFps(VideoPath, SampleFrames=600) -> dict:
    """The camera reports 150 fps; check how many consecutive frames are
    exact repeats, which would mean the real rate is lower."""
    Cap = cv2.VideoCapture(str(VideoPath))
    ReportedFps = Cap.get(cv2.CAP_PROP_FPS)
    Previous, Repeats, Count = None, 0, 0
    while Count < SampleFrames:
        Ok, Frame = Cap.read()
        if not Ok:
            break
        Small = cv2.resize(cv2.cvtColor(Frame, cv2.COLOR_BGR2GRAY), (160, 90)).astype(np.int16)
        if Previous is not None and np.abs(Small - Previous).mean() < 0.5:
            Repeats += 1
        Previous, Count = Small, Count + 1
    Cap.release()
    UniqueFraction = 1 - Repeats / max(Count - 1, 1)
    return {
        "ReportedFps": round(ReportedFps, 2),
        "RepeatedFrameFraction": round(1 - UniqueFraction, 3),
        "EstimatedRealFps": round(ReportedFps * UniqueFraction, 1),
    }


def _TrackerConfig(TrackBufferFrames: int) -> str:
    """Ultralytics' ByteTrack defaults with a longer memory for lost tracks,
    so a cow hidden for a few seconds keeps her TrackId."""
    import ultralytics

    Default = Path(ultralytics.__file__).parent / "cfg" / "trackers" / "bytetrack.yaml"
    Text = re.sub(r"(?m)^track_buffer:\s*\d+", f"track_buffer: {TrackBufferFrames}", Default.read_text())
    Fd, TrackerFile = tempfile.mkstemp(suffix=".yaml", prefix="ByteTrackCows")
    with os.fdopen(Fd, "w") as F:
        F.write(Text)
    return TrackerFile


def _Color(TrackId: int) -> tuple:
    Rng = np.random.default_rng(TrackId)
    return tuple(int(C) for C in Rng.integers(64, 256, 3))


def _DrawTracks(Frame, Boxes, Scale, TimeSeconds):
    Small = cv2.resize(Frame, None, fx=Scale, fy=Scale)
    for TrackId, X1, Y1, X2, Y2, Confidence in Boxes:
        P1, P2 = (int(X1 * Scale), int(Y1 * Scale)), (int(X2 * Scale), int(Y2 * Scale))
        Color = _Color(TrackId)
        cv2.rectangle(Small, P1, P2, Color, 2)
        cv2.putText(Small, f"{TrackId} {Confidence:.2f}", (P1[0], max(P1[1] - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, Color, 2)
    cv2.putText(Small, f"{TimeSeconds:7.1f}s", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return Small


def SummarizeTracks(Tracks: pd.DataFrame) -> pd.DataFrame:
    if Tracks.empty:
        return pd.DataFrame(columns=["TrackId", "FirstSeconds", "LastSeconds", "Frames",
                                     "MeanConfidence", "MeanBoxArea"])
    Tracks = Tracks.assign(BoxArea=(Tracks.X2 - Tracks.X1) * (Tracks.Y2 - Tracks.Y1))
    return (Tracks.groupby("TrackId")
            .agg(FirstSeconds=("TimeSeconds", "min"), LastSeconds=("TimeSeconds", "max"),
                 Frames=("Frame", "count"), MeanConfidence=("Confidence", "mean"),
                 MeanBoxArea=("BoxArea", "mean"))
            .round(2).reset_index().sort_values("FirstSeconds"))


def DetectAndTrack(VideoPath, OutputDir, TargetFps=15, ModelName="yolo26n.pt", ClassIds=(CocoCow,),
                   Confidence=0.25, ImageSize=640, TrackBufferSeconds=10, MaxSeconds=None,
                   PreviewSeconds=60, PreviewWidth=960) -> dict:
    """Run detection + tracking on every n-th frame so about ``TargetFps``
    frames per second are processed (30 fps clip, 15 target = every 2nd)."""
    from ultralytics import YOLO

    OutputDir = Path(OutputDir)
    OutputDir.mkdir(parents=True, exist_ok=True)
    Info = ProbeVideo(VideoPath)
    Fps = Info["Fps"]
    Stride = max(1, round(Fps / TargetFps))
    LastFrame = int(MaxSeconds * Fps) if MaxSeconds else Info["FrameCount"]
    print(f"{Info['Video']}: {Info['Width']}x{Info['Height']} at {Fps} fps, "
          f"every {Stride}th frame (~{Fps / Stride:.1f} fps), up to {LastFrame / Fps:.0f} s")

    Model = YOLO(ModelName)
    TrackerPath = _TrackerConfig(int(TrackBufferSeconds * Fps / Stride))
    Scale = PreviewWidth / Info["Width"]
    PreviewTemp = OutputDir / "PreviewTracks.tmp.avi"
    Preview = None

    Cap = cv2.VideoCapture(str(VideoPath))
    Started, Processed, FrameIndex = time.time(), 0, -1
    try:
        with (OutputDir / "Tracks.csv").open("w", newline="") as F:
            Writer = csv.writer(F)
            Writer.writerow(TrackColumns)
            while FrameIndex + 1 < LastFrame:
                FrameIndex += 1
                if not Cap.grab():
                    break
                if FrameIndex % Stride:
                    continue
                Ok, Frame = Cap.retrieve()
                if not Ok:
                    continue
                TimeSeconds = round(FrameIndex / Fps, 3)
                Result = Model.track(Frame, persist=True, tracker=TrackerPath,
                                     classes=list(ClassIds) if ClassIds else None,
                                     conf=Confidence, imgsz=ImageSize, verbose=False)[0]
                Boxes = []
                if Result.boxes is not None and Result.boxes.id is not None:
                    for (X1, Y1, X2, Y2), TrackId, Conf, ClassId in zip(
                            Result.boxes.xyxy.cpu().numpy().round(1), Result.boxes.id.int().tolist(),
                            Result.boxes.conf.tolist(), Result.boxes.cls.int().tolist()):
                        Writer.writerow([FrameIndex, TimeSeconds, TrackId, X1, Y1, X2, Y2, round(Conf, 3), ClassId])
                        Boxes.append((TrackId, X1, Y1, X2, Y2, Conf))

                if TimeSeconds < PreviewSeconds:
                    Drawn = _DrawTracks(Frame, Boxes, Scale, TimeSeconds)
                    if Preview is None:
                        Preview = cv2.VideoWriter(str(PreviewTemp), cv2.VideoWriter_fourcc(*"MJPG"),
                                                  Fps / Stride, (Drawn.shape[1], Drawn.shape[0]))
                    Preview.write(Drawn)

                Processed += 1
                if Processed % 500 == 0:
                    print(f"  {TimeSeconds:.0f} s done, {Processed / (time.time() - Started):.1f} frames/s")
    finally:
        Cap.release()
        if Preview is not None:
            Preview.release()
        os.remove(TrackerPath)
        del Model
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    if PreviewTemp.exists():
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(PreviewTemp), "-c:v", "libx264",
                        "-preset", "veryfast", "-pix_fmt", "yuv420p", str(OutputDir / "PreviewTracks.mp4")],
                       check=True)
        PreviewTemp.unlink()

    Tracks = pd.read_csv(OutputDir / "Tracks.csv")
    Summary = SummarizeTracks(Tracks)
    Summary.to_csv(OutputDir / "TrackSummary.csv", index=False)
    Seconds = time.time() - Started
    Stats = {
        "ProcessedFrames": Processed,
        "Stride": Stride,
        "Seconds": round(Seconds, 1),
        "FramesPerSecond": round(Processed / Seconds, 1) if Seconds else 0,
        "Boxes": len(Tracks),
        "Tracks": len(Summary),
        "FramesWithCow": int(Tracks.Frame.nunique()) if len(Tracks) else 0,
    }
    print(Stats)
    return Stats
