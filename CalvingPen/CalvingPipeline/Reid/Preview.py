"""A preview video of the tracking step (``Segments.TrackAndCrop``), made
afterwards from its ``Tracks.csv``: the same frames, the YOLO + ByteTrack boxes
with their TrackIds, and the SAM 2.1 mask of every boxed cow (run again here on
every preview frame; the tracking step only masks every ``EverySeconds``).
Frames where the tracking step saved crops are marked "CROP".

``PreviewVideo(VideoPath, VideoDir)`` writes ``VideoDir/PreviewTracks.mp4``
(overwritten). By default it shows the ``Minutes`` with the most tracks.
"""

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .Frames import LocalCopy, SampleFrames
from .Segments import _Masks


def _Color(TrackId: int) -> tuple:
    Rng = np.random.default_rng(int(TrackId))
    return tuple(int(C) for C in Rng.integers(64, 256, 3))


def BusiestStart(Tracks, Minutes=5) -> float:
    """Start (s) of the ``Minutes`` window with the most different TrackIds."""
    if Tracks.empty:
        return 0.0
    Bin = (Tracks.TimeSeconds // 60).astype(int)
    PerMinute = Tracks.groupby(Bin).TrackId.apply(set)
    Best, BestCount = PerMinute.index.min(), -1
    for M in PerMinute.index:
        Ids = set().union(*[PerMinute.get(K, set()) for K in range(M, M + Minutes)])
        if len(Ids) > BestCount:
            Best, BestCount = M, len(Ids)
    return float(Best * 60)


def DrawFrame(Frame, Boxes, Masks, TimeSeconds, Width=960, Crop=False):
    """Masks (coloured, half see-through), boxes and TrackIds, scaled to ``Width``."""
    Out = Frame.copy()
    for (Id, *_), Mask in zip(Boxes, Masks if Masks is not None else [None] * len(Boxes)):
        if Mask is not None:
            Out[Mask] = (0.5 * Out[Mask] + 0.5 * np.array(_Color(Id))).astype(np.uint8)
    Scale = Width / Frame.shape[1]
    Out = cv2.resize(Out, None, fx=Scale, fy=Scale, interpolation=cv2.INTER_AREA)
    for Id, X1, Y1, X2, Y2, Conf in Boxes:
        P1, P2 = (int(X1 * Scale), int(Y1 * Scale)), (int(X2 * Scale), int(Y2 * Scale))
        cv2.rectangle(Out, P1, P2, _Color(Id), 2)
        cv2.putText(Out, f"T{Id} {Conf:.2f}", (P1[0] + 3, max(P1[1] - 6, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    _Color(Id), 2)
    H, M, S = int(TimeSeconds // 3600), int(TimeSeconds % 3600 // 60), TimeSeconds % 60
    cv2.putText(Out, f"{H}:{M:02d}:{S:04.1f}  {len(Boxes)} cows" + ("  CROP" if Crop else ""), (8, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return Out


def PreviewVideo(VideoPath, VideoDir, StartSeconds=None, Minutes=5, Speed=4, Width=960, WithMasks=True,
                 SegModel=None) -> Path:
    """Preview of ``Minutes`` from ``StartSeconds`` (None = the busiest
    window), played ``Speed`` times faster. Needs the finished (or flushed)
    ``Tracks.csv`` of ``TrackAndCrop`` in ``VideoDir``."""
    VideoDir = Path(VideoDir)
    Settings = json.loads((VideoDir / "Run.json").read_text())["Settings"]
    Tracks = pd.read_csv(VideoDir / "Tracks.csv")
    Fps, Origin = Settings["TargetFps"], float(Settings.get("StartSeconds") or 0)

    def Step(Seconds):   # sample number on the tracking step's time grid (a Tick repeats while a picture is held)
        return np.round((np.asarray(Seconds, float) - Origin) * Fps).astype(int)

    CropSteps = (set(Step(pd.read_csv(VideoDir / "Crops.csv", usecols=["TimeSeconds"]).TimeSeconds).tolist())
                 if (VideoDir / "Crops.csv").exists() else set())
    if StartSeconds is None:
        StartSeconds = BusiestStart(Tracks, Minutes)
    StartSeconds = Origin + max(0, int(Step(StartSeconds))) / Fps     # on the same grid as Tracks.csv
    EndSeconds = StartSeconds + Minutes * 60
    Near = Tracks[(Tracks.TimeSeconds >= StartSeconds - 1) & (Tracks.TimeSeconds <= EndSeconds + 1)]
    ByStep = {K: G for K, G in Near.groupby(Step(Near.TimeSeconds))}
    Sam = None
    if WithMasks:
        from ultralytics import SAM
        Sam = SAM(SegModel or Settings["SegModel"])
    Temp, Out = VideoDir / "PreviewTracks.tmp.avi", VideoDir / "PreviewTracks.mp4"
    Writer, Count = None, 0
    print(f"Preview of {Settings['Video']}: {StartSeconds / 60:.1f}-{EndSeconds / 60:.1f} min, "
          f"{Fps} fps shown at {Speed}x" + (", with SAM masks" if WithMasks else ""))
    try:
        for _, T, Frame in SampleFrames(LocalCopy(VideoPath), Fps, StartSeconds, EndSeconds, Settings["NativeFps"]):
            K = int(Step(T))
            Rows = ByStep.get(K)
            Boxes = [] if Rows is None else [(int(R.TrackId), R.X1, R.Y1, R.X2, R.Y2, R.Confidence)
                                              for R in Rows.itertuples()]
            Masks = _Masks(Sam, Frame, [B[1:5] for B in Boxes]) if (Sam is not None and Boxes) else None
            Drawn = DrawFrame(Frame, Boxes, Masks, T, Width, Crop=K in CropSteps)
            if Writer is None:
                Writer = cv2.VideoWriter(str(Temp), cv2.VideoWriter_fourcc(*"MJPG"), Fps * Speed,
                                         (Drawn.shape[1], Drawn.shape[0]))
            Writer.write(Drawn)
            Count += 1
    finally:
        if Writer is not None:
            Writer.release()
        del Sam
    if not Count:
        raise ValueError(f"No frames between {StartSeconds:.0f} and {EndSeconds:.0f} s")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(Temp), "-c:v", "libx264", "-preset", "veryfast",
                    "-pix_fmt", "yuv420p", str(Out)], check=True)
    Temp.unlink()
    print(f"{Count} frames -> {Out}")
    return Out
