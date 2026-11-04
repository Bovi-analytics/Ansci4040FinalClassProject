"""Step 02c (experiment): track the cow with SAM 2.1's video mode instead of
ByteTrack, on one time window, and compare the two.

ByteTrack (step 02b) links YOLO boxes by position only, so whenever YOLO
misses the cow for > 10 s (lying, straining) she gets a new TrackId. SAM 2.1
in video mode follows each object's *mask* using a memory of how it looked,
so it may keep following her where YOLO fails. Here:

1. the window is cut into chunks of ``ChunkSeconds`` at ``TargetFps``
   (SAM 2's memory grows with every frame, so an hour cannot go in one go),
2. YOLO26n runs on every chunk frame; the first frame with a confident cow
   box seeds SAM 2 (one object per box, duplicates removed),
3. SAM 2.1 Small follows every seeded object through the chunk,
4. at each chunk border an object continues the previous chunk's track if
   its first box overlaps that track's last box (IoU >= ``LinkIou``),
5. every SAM box is compared with YOLO's boxes in the same frame:
   ``YoloIoU`` high = YOLO agrees; no YOLO box = SAM is covering a frame
   YOLO missed (the point of this experiment); YOLO boxes elsewhere but none
   on SAM's = SAM may have drifted (to the calf, the floor, ...).

Outputs in ``OutputDir``, overwritten on every run; ``Tracks.csv`` has the
same columns as step 02b (plus YOLO-agreement columns), so step 03 can use it:

- ``Tracks.csv``, ``TrackSummary.csv``
- ``PreviewTracks.mp4``: masks + boxes + IDs, sped up ``PreviewSpeed`` times
- ``Comparison.csv``: ByteTrack (02b) vs SAM 2 on the same window
"""

import csv
import gc
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .CropCows import RemoveDuplicateBoxes
from .DetectTrack import CocoCow, _Color
from .VideoTools import FormatTime, ProbeVideo

TrackColumns = ["Frame", "TimeSeconds", "TrackId", "X1", "Y1", "X2", "Y2", "Confidence", "ClassId",
                "MaskArea", "YoloIoU", "YoloConfidence", "Chunk"]


def BoxIou(A, B) -> float:
    W = min(A[2], B[2]) - max(A[0], B[0])
    H = min(A[3], B[3]) - max(A[1], B[1])
    if W <= 0 or H <= 0:
        return 0.0
    Inter = W * H
    return Inter / ((A[2] - A[0]) * (A[3] - A[1]) + (B[2] - B[0]) * (B[3] - B[1]) - Inter)


def _CutChunk(ClipPath, StartSeconds, Seconds, TargetFps, OutPath):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{StartSeconds:.3f}", "-i", str(ClipPath),
                    "-t", f"{Seconds:.3f}", "-vf", f"fps={TargetFps}", "-c:v", "libx264", "-preset", "veryfast",
                    "-crf", "18", "-pix_fmt", "yuv420p", "-an", str(OutPath)], check=True)


def _YoloBoxes(Yolo, ChunkPath, ClassIds, Confidence, ImageSize):
    """YOLO boxes for every frame of a chunk: list of (N, 5) arrays [x1, y1, x2, y2, conf]."""
    Boxes, Cap = [], cv2.VideoCapture(str(ChunkPath))
    while True:
        Ok, Frame = Cap.read()
        if not Ok:
            break
        R = Yolo.predict(Frame, classes=list(ClassIds) if ClassIds else None, conf=Confidence,
                         imgsz=ImageSize, verbose=False)[0]
        if R.boxes is None or len(R.boxes) == 0:
            Boxes.append(np.zeros((0, 5)))
        else:
            Boxes.append(np.c_[R.boxes.xyxy.cpu().numpy(), R.boxes.conf.cpu().numpy()])
    Cap.release()
    return Boxes


def _Seeds(Boxes: np.ndarray, SeedConfidence) -> np.ndarray:
    Confident = Boxes[Boxes[:, 4] >= SeedConfidence]
    if len(Confident) < 2:
        return Confident
    Frame = pd.DataFrame(Confident, columns=["X1", "Y1", "X2", "Y2", "Confidence"]).assign(Frame=0)
    return RemoveDuplicateBoxes(Frame)[["X1", "Y1", "X2", "Y2", "Confidence"]].to_numpy()


def _Draw(Frame, Objects, Scale, TimeSeconds, Chunk):
    Small = cv2.resize(Frame, None, fx=Scale, fy=Scale)
    Overlay = Small.copy()
    for TrackId, Box, Mask, Status in Objects:
        Color = _Color(TrackId)
        M = cv2.resize(Mask.astype(np.uint8), (Small.shape[1], Small.shape[0])) > 0
        Overlay[M] = Color
        P1, P2 = (int(Box[0] * Scale), int(Box[1] * Scale)), (int(Box[2] * Scale), int(Box[3] * Scale))
        cv2.rectangle(Small, P1, P2, Color, 2)
        cv2.putText(Small, f"{TrackId} {Status}", (P1[0], max(P1[1] - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, Color, 2)
    Small = cv2.addWeighted(Overlay, 0.35, Small, 0.65, 0)
    cv2.putText(Small, f"{TimeSeconds:7.1f}s  chunk {Chunk}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return Small


def _TrackStats(Tracks: pd.DataFrame, WindowFrames, MinFrames) -> dict:
    if Tracks.empty:
        return {"Tracks": 0, f"TracksWith{MinFrames}PlusFrames": 0, "FramesWithCowShare": 0.0,
                "LongestTrackSeconds": 0.0, "MedianTrackSeconds": 0.0}
    Spans = Tracks.groupby("TrackId").TimeSeconds.agg(lambda T: T.max() - T.min())
    Counts = Tracks.groupby("TrackId").size()
    return {"Tracks": int(Tracks.TrackId.nunique()),
            f"TracksWith{MinFrames}PlusFrames": int((Counts >= MinFrames).sum()),
            "FramesWithCowShare": round(Tracks.TimeSeconds.round(1).nunique() / max(WindowFrames, 1), 3),
            "LongestTrackSeconds": round(float(Spans.max()), 1),
            "MedianTrackSeconds": round(float(Spans.median()), 1)}


def Sam2TrackWindow(ClipPath, OutputDir, StartSeconds, EndSeconds, TargetFps=5, ChunkSeconds=120,
                    DetModel="yolo26n.pt", ClassIds=(CocoCow,), Confidence=0.25, SeedConfidence=0.4,
                    SegModel="sam2.1_s.pt", ImageSize=1024, LinkIou=0.3, CompareTracksPath=None,
                    MinFrames=25, PreviewSpeed=4, PreviewWidth=960, WorkDir=None) -> dict:
    from ultralytics import YOLO
    from ultralytics.models.sam import SAM2VideoPredictor

    OutputDir = Path(OutputDir)
    OutputDir.mkdir(parents=True, exist_ok=True)
    WorkDir = Path(WorkDir or tempfile.mkdtemp(prefix="Sam2Chunks"))   # local disk: much faster than Drive
    WorkDir.mkdir(parents=True, exist_ok=True)
    Info = ProbeVideo(ClipPath)
    ClipFps = Info["Fps"]
    EndSeconds = min(EndSeconds, Info["DurationSeconds"])
    Scale = PreviewWidth / Info["Width"]
    print(f"{Info['Video']}: {FormatTime(StartSeconds)}-{FormatTime(EndSeconds)} of the clip, "
          f"{TargetFps} fps, chunks of {ChunkSeconds} s")

    Yolo = YOLO(DetModel)
    PreviewTemp = OutputDir / "PreviewTracks.tmp.avi"
    Preview, NextTrackId, Previous = None, 1, []      # Previous: (TrackId, last box) of the last chunk
    Started, Processed, Chunks = time.time(), 0, 0

    with (OutputDir / "Tracks.csv").open("w", newline="") as F:
        Writer = csv.writer(F)
        Writer.writerow(TrackColumns)
        ChunkStart = StartSeconds
        while ChunkStart < EndSeconds - 1e-6:
            Chunks += 1
            Seconds = min(ChunkSeconds, EndSeconds - ChunkStart)
            ChunkPath = WorkDir / f"Chunk{Chunks:03d}.mp4"
            _CutChunk(ClipPath, ChunkStart, Seconds, TargetFps, ChunkPath)
            YoloAll = _YoloBoxes(Yolo, ChunkPath, ClassIds, Confidence, ImageSize)
            Processed += len(YoloAll)
            SeedIndex = next((I for I, B in enumerate(YoloAll) if (B[:, 4] >= SeedConfidence).any()), None)
            if SeedIndex is None:
                print(f"  chunk {Chunks} ({FormatTime(ChunkStart)}): no confident cow box, skipped")
                Previous, ChunkStart = [], ChunkStart + Seconds
                continue
            Seeds = _Seeds(YoloAll[SeedIndex], SeedConfidence)
            SeedStart = ChunkStart + SeedIndex / TargetFps
            if SeedIndex > 0:   # SAM 2 takes its prompts on the first frame: start the chunk at the seed
                _CutChunk(ClipPath, SeedStart, ChunkStart + Seconds - SeedStart, TargetFps, ChunkPath)

            Predictor = SAM2VideoPredictor(overrides=dict(conf=0.25, task="segment", mode="predict", imgsz=ImageSize,
                                                          model=SegModel, save=False, verbose=False))
            Ids, LastBox = {}, {}
            for I, R in enumerate(Predictor(source=str(ChunkPath), bboxes=Seeds[:, :4].tolist(), stream=True)):
                TimeSeconds = round(SeedStart + I / TargetFps, 3)
                FrameYolo = YoloAll[SeedIndex + I] if SeedIndex + I < len(YoloAll) else np.zeros((0, 5))
                Objects = []
                if R.boxes is not None and len(R.boxes):
                    Masks = R.masks.data.cpu().numpy().astype(bool)
                    for Box, Obj, Mask in zip(R.boxes.xyxy.cpu().numpy(), R.boxes.cls.int().tolist(), Masks):
                        if Obj not in Ids:   # first sighting: continue a track from the previous chunk?
                            Match = max(Previous, key=lambda P: BoxIou(P[1], Box), default=None)
                            if Match is not None and BoxIou(Match[1], Box) >= LinkIou and Match[0] not in Ids.values():
                                Ids[Obj] = Match[0]
                            else:
                                Ids[Obj], NextTrackId = NextTrackId, NextTrackId + 1
                        Ious = [BoxIou(Box, Y) for Y in FrameYolo]
                        Best = int(np.argmax(Ious)) if Ious else None
                        YoloIou = round(Ious[Best], 3) if Ious else 0.0
                        YoloConf = round(float(FrameYolo[Best, 4]), 3) if Ious else 0.0
                        Status = "ok" if YoloIou >= 0.3 else ("??" if len(FrameYolo) else "sam")
                        Writer.writerow([int(round(TimeSeconds * ClipFps)), TimeSeconds, Ids[Obj], *np.round(Box, 1),
                                         1.0, CocoCow, int(Mask.sum()), YoloIou, YoloConf, Chunks])
                        LastBox[Ids[Obj]] = Box
                        Objects.append((Ids[Obj], Box, Mask, Status))
                Drawn = _Draw(R.orig_img, Objects, Scale, TimeSeconds, Chunks)
                if Preview is None:
                    Preview = cv2.VideoWriter(str(PreviewTemp), cv2.VideoWriter_fourcc(*"MJPG"),
                                              TargetFps * PreviewSpeed, (Drawn.shape[1], Drawn.shape[0]))
                Preview.write(Drawn)
            del Predictor
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            Previous = list(LastBox.items())
            print(f"  chunk {Chunks} ({FormatTime(ChunkStart)}): {len(Seeds)} seed(s), "
                  f"{Processed / (time.time() - Started):.1f} frames/s overall")
            ChunkStart += Seconds

    if Preview is not None:
        Preview.release()
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(PreviewTemp), "-c:v", "libx264", "-preset", "veryfast",
                        "-pix_fmt", "yuv420p", str(OutputDir / "PreviewTracks.mp4")], check=True)
        PreviewTemp.unlink()
    shutil.rmtree(WorkDir, ignore_errors=True)
    del Yolo
    gc.collect()

    Tracks = pd.read_csv(OutputDir / "Tracks.csv")
    if len(Tracks):
        Tracks["YoloStatus"] = np.where(Tracks.YoloIoU >= 0.3, "Agrees",
                                        np.where(Tracks.YoloConfidence > 0, "Disagrees", "NoYoloBox"))
        Summary = (Tracks.groupby("TrackId")
                   .agg(FirstSeconds=("TimeSeconds", "min"), LastSeconds=("TimeSeconds", "max"),
                        Frames=("Frame", "count"), MedianMaskArea=("MaskArea", "median"),
                        YoloAgrees=("YoloStatus", lambda S: (S == "Agrees").mean()),
                        NoYoloBox=("YoloStatus", lambda S: (S == "NoYoloBox").mean()),
                        YoloDisagrees=("YoloStatus", lambda S: (S == "Disagrees").mean()),
                        MeanConfidence=("Confidence", "mean"))
                   .round(3).reset_index())
    else:
        Summary = pd.DataFrame(columns=["TrackId", "FirstSeconds", "LastSeconds", "Frames"])
    Summary.to_csv(OutputDir / "TrackSummary.csv", index=False)

    WindowFrames = int((EndSeconds - StartSeconds) * TargetFps)
    Rows = [{"Method": "SAM 2.1 video (02c)", **_TrackStats(Tracks, WindowFrames, MinFrames)}]
    if CompareTracksPath and Path(CompareTracksPath).exists():
        Byte = pd.read_csv(CompareTracksPath)
        Byte = Byte[(Byte.TimeSeconds >= StartSeconds) & (Byte.TimeSeconds < EndSeconds)]
        Rows.insert(0, {"Method": "ByteTrack (02b)", **_TrackStats(Byte, WindowFrames, MinFrames)})
    Comparison = pd.DataFrame(Rows)
    Comparison.to_csv(OutputDir / "Comparison.csv", index=False)

    Stats = {"Chunks": Chunks, "ProcessedFrames": Processed, "Seconds": round(time.time() - Started, 1)}
    if len(Tracks):
        Stats.update({S: round(float((Tracks.YoloStatus == S).mean()), 3) for S in ("Agrees", "NoYoloBox", "Disagrees")})
    print(Stats)
    return Stats
