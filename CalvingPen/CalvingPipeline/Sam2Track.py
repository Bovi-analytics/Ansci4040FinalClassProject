"""Step 02c / 02d: follow the cow with SAM 2.1's video mode (replaces ByteTrack).

ByteTrack (step 02b) links YOLO boxes by position only, so whenever YOLO
misses the cow for > 10 s (lying, straining) she gets a new TrackId: one cow
became 47 tracks. SAM 2.1 in video mode follows each object's *mask* using a
memory of how it looked, so it keeps following her where YOLO fails. On the
reference clip (1080-1980 s) it kept her as **one track for 15 minutes**,
and YOLO agreed with SAM's box on 97 % of frames.

How:

1. the clip (or a window of it) is cut into chunks of ``ChunkSeconds`` at
   ``TargetFps`` (SAM 2's memory grows with every frame, so an hour cannot go
   in one go). 2 fps is enough: the camera only updates ~2 times a second,
2. YOLO26n runs on every chunk frame; the first frame with a confident cow
   box seeds SAM 2 (one object per box, duplicates removed),
3. SAM 2.1 Small follows every seeded object through the chunk,
4. at each chunk border an object continues the previous chunk's track if
   its first box overlaps that track's last box (IoU >= ``LinkIou``),
5. every SAM box is compared with YOLO's boxes in the same frame
   (``YoloStatus``): **Agrees** = YOLO sees her in the same place;
   **NoYoloBox** = YOLO saw nothing, SAM kept following her; **Disagrees** =
   YOLO sees a cow elsewhere but not where SAM is (SAM may have drifted).

**Resumable:** every finished chunk is saved in ``OutputDir/Chunks/``. If
the run stops (Colab disconnects, GPU time runs out), run it again with the
same settings and it continues from the first unfinished chunk. Changing a
setting, or ``Resume=False``, starts over.

Outputs in ``OutputDir`` (rebuilt from the chunks at the end of every run);
``Tracks.csv`` has the same columns as step 02b plus the YOLO-agreement
columns, so step 03 reads it the same way:

- ``Tracks.csv``, ``TrackSummary.csv``
- ``PreviewTracks.mp4``: masks + boxes + IDs, sped up ``PreviewSpeed`` times
  (``Preview=False`` skips it and saves time)
- ``Comparison.csv``: ByteTrack (02b) vs SAM 2 on the same time span
- ``Chunks/``: the per-chunk pieces (keep them; they make resuming work)
"""

import csv
import gc
import json
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


def _Encode(SourcePath, OutPath):
    """Same settings for every preview piece, so they can be joined without re-encoding."""
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(SourcePath), "-c:v", "libx264", "-preset", "veryfast",
                    "-crf", "28", "-pix_fmt", "yuv420p", "-an", str(OutPath)], check=True)


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


def _FreeGpu():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


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


def SummarizeSamTracks(Tracks: pd.DataFrame) -> pd.DataFrame:
    if Tracks.empty:
        return pd.DataFrame(columns=["TrackId", "FirstSeconds", "LastSeconds", "Frames"])
    Status = np.where(Tracks.YoloIoU >= 0.3, "Agrees", np.where(Tracks.YoloConfidence > 0, "Disagrees", "NoYoloBox"))
    Tracks = Tracks.assign(YoloStatus=Status)
    return (Tracks.groupby("TrackId")
            .agg(FirstSeconds=("TimeSeconds", "min"), LastSeconds=("TimeSeconds", "max"),
                 Frames=("Frame", "count"), MedianMaskArea=("MaskArea", "median"),
                 YoloAgrees=("YoloStatus", lambda S: (S == "Agrees").mean()),
                 NoYoloBox=("YoloStatus", lambda S: (S == "NoYoloBox").mean()),
                 YoloDisagrees=("YoloStatus", lambda S: (S == "Disagrees").mean()),
                 MeanConfidence=("Confidence", "mean"))
            .round(3).reset_index().sort_values("FirstSeconds"))


def _TrackChunk(Yolo, ClipPath, ChunkStart, Seconds, Number, Previous, NextTrackId, Settings, WorkDir,
                ChunkDir, Scale, Preview):
    """Track one chunk; writes ``Chunk<n>.csv`` (+ preview piece) and returns the
    state the next chunk needs: (last box per track, next free TrackId)."""
    from ultralytics.models.sam import SAM2VideoPredictor

    S = Settings
    ChunkPath = WorkDir / f"Chunk{Number:03d}.mp4"
    _CutChunk(ClipPath, ChunkStart, Seconds, S["TargetFps"], ChunkPath)
    YoloAll = _YoloBoxes(Yolo, ChunkPath, S["ClassIds"], S["Confidence"], S["ImageSize"])
    SeedIndex = next((I for I, B in enumerate(YoloAll) if (B[:, 4] >= S["SeedConfidence"]).any()), None)
    Rows, LastBox, PreviewWriter, Seeds = [], {}, None, []
    PreviewTemp = WorkDir / f"Preview{Number:03d}.avi"

    if SeedIndex is not None:
        Seeds = _Seeds(YoloAll[SeedIndex], S["SeedConfidence"])
        SeedStart = ChunkStart + SeedIndex / S["TargetFps"]
        if SeedIndex > 0:   # SAM 2 takes its prompts on the first frame: start the chunk at the seed
            _CutChunk(ClipPath, SeedStart, ChunkStart + Seconds - SeedStart, S["TargetFps"], ChunkPath)
        Predictor = SAM2VideoPredictor(overrides=dict(conf=0.25, task="segment", mode="predict", imgsz=S["ImageSize"],
                                                      model=S["SegModel"], save=False, verbose=False))
        Ids = {}
        for I, R in enumerate(Predictor(source=str(ChunkPath), bboxes=Seeds[:, :4].tolist(), stream=True)):
            TimeSeconds = round(SeedStart + I / S["TargetFps"], 3)
            FrameYolo = YoloAll[SeedIndex + I] if SeedIndex + I < len(YoloAll) else np.zeros((0, 5))
            Objects = []
            if R.boxes is not None and len(R.boxes):
                Masks = R.masks.data.cpu().numpy().astype(bool)
                for Box, Obj, Mask in zip(R.boxes.xyxy.cpu().numpy(), R.boxes.cls.int().tolist(), Masks):
                    if Obj not in Ids:   # first sighting: continue a track from the previous chunk?
                        Match = max(Previous, key=lambda P: BoxIou(P[1], Box), default=None)
                        if Match is not None and BoxIou(Match[1], Box) >= S["LinkIou"] and Match[0] not in Ids.values():
                            Ids[Obj] = Match[0]
                        else:
                            Ids[Obj], NextTrackId = NextTrackId, NextTrackId + 1
                    Ious = [BoxIou(Box, Y) for Y in FrameYolo]
                    Best = int(np.argmax(Ious)) if Ious else None
                    YoloIou = round(Ious[Best], 3) if Ious else 0.0
                    YoloConf = round(float(FrameYolo[Best, 4]), 3) if Ious else 0.0
                    Status = "ok" if YoloIou >= 0.3 else ("??" if len(FrameYolo) else "sam")
                    Rows.append([int(round(TimeSeconds * S["ClipFps"])), TimeSeconds, Ids[Obj],
                                 *[float(V) for V in np.round(Box, 1)], 1.0, CocoCow, int(Mask.sum()),
                                 YoloIou, YoloConf, Number])
                    LastBox[Ids[Obj]] = [float(V) for V in Box]
                    Objects.append((Ids[Obj], Box, Mask, Status))
            if Preview:
                Drawn = _Draw(R.orig_img, Objects, Scale, TimeSeconds, Number)
                if PreviewWriter is None:
                    PreviewWriter = cv2.VideoWriter(str(PreviewTemp), cv2.VideoWriter_fourcc(*"MJPG"),
                                                    S["TargetFps"] * S["PreviewSpeed"], (Drawn.shape[1], Drawn.shape[0]))
                PreviewWriter.write(Drawn)
        del Predictor
        _FreeGpu()

    if PreviewWriter is not None:
        PreviewWriter.release()
        _Encode(PreviewTemp, ChunkDir / f"Preview{Number:03d}.mp4")
        PreviewTemp.unlink()
    ChunkPath.unlink(missing_ok=True)

    with (ChunkDir / f"Chunk{Number:03d}.csv").open("w", newline="") as F:
        Writer = csv.writer(F)
        Writer.writerow(TrackColumns)
        Writer.writerows(Rows)
    State = {"Chunk": Number, "StartSeconds": ChunkStart, "Seconds": Seconds, "Seeds": len(Seeds),
             "Previous": [[T, B] for T, B in LastBox.items()], "NextTrackId": NextTrackId}
    # Written last: a chunk only counts as done once its .json exists.
    (ChunkDir / f"Chunk{Number:03d}.json").write_text(json.dumps(State))
    return State


def Sam2Track(ClipPath, OutputDir, StartSeconds=0, EndSeconds=None, TargetFps=2, ChunkSeconds=120,
              DetModel="yolo26n.pt", ClassIds=(CocoCow,), Confidence=0.25, SeedConfidence=0.4,
              SegModel="sam2.1_s.pt", ImageSize=1024, LinkIou=0.3, CompareTracksPath=None,
              MinFrames=10, Preview=True, PreviewSpeed=4, PreviewWidth=960, Resume=True, WorkDir=None) -> dict:
    """Track ``StartSeconds``..``EndSeconds`` of the clip (None = to the end)."""
    from ultralytics import YOLO

    OutputDir = Path(OutputDir)
    ChunkDir = OutputDir / "Chunks"
    OutputDir.mkdir(parents=True, exist_ok=True)
    Info = ProbeVideo(ClipPath)
    EndSeconds = min(EndSeconds or Info["DurationSeconds"], Info["DurationSeconds"])
    Settings = {"Clip": str(ClipPath), "StartSeconds": StartSeconds, "EndSeconds": EndSeconds,
                "TargetFps": TargetFps, "ChunkSeconds": ChunkSeconds, "DetModel": DetModel,
                "ClassIds": list(ClassIds) if ClassIds else None, "Confidence": Confidence,
                "SeedConfidence": SeedConfidence, "SegModel": SegModel, "ImageSize": ImageSize,
                "LinkIou": LinkIou, "Preview": Preview, "PreviewSpeed": PreviewSpeed, "PreviewWidth": PreviewWidth,
                "ClipFps": Info["Fps"]}

    SettingsPath = ChunkDir / "Settings.json"
    Same = SettingsPath.exists() and json.loads(SettingsPath.read_text()) == Settings
    if not (Resume and Same):
        if ChunkDir.exists():
            print("Starting over" + ("" if Resume else " (Resume=False)") +
                  ("" if Same or not SettingsPath.exists() else ": settings changed"))
        shutil.rmtree(ChunkDir, ignore_errors=True)
    ChunkDir.mkdir(exist_ok=True)
    (OutputDir / "PreviewTracks.tmp.avi").unlink(missing_ok=True)   # left over by the first, non-resumable version
    SettingsPath.write_text(json.dumps(Settings, indent=1))

    Starts = list(np.arange(StartSeconds, EndSeconds - 1e-6, ChunkSeconds).round(3))
    Done = [N for N in range(1, len(Starts) + 1) if (ChunkDir / f"Chunk{N:03d}.json").exists()]
    FirstTodo = next((N for N in range(1, len(Starts) + 1) if N not in Done), None)
    print(f"{Info['Video']}: {FormatTime(StartSeconds)}-{FormatTime(EndSeconds)} of the clip, {TargetFps} fps, "
          f"{len(Starts)} chunks of {ChunkSeconds} s; {len(Done)} already done")

    Started = time.time()
    if FirstTodo is not None:
        if FirstTodo > 1:
            State = json.loads((ChunkDir / f"Chunk{FirstTodo - 1:03d}.json").read_text())
            Previous, NextTrackId = [(T, B) for T, B in State["Previous"]], State["NextTrackId"]
        else:
            Previous, NextTrackId = [], 1
        WorkDir = Path(WorkDir or tempfile.mkdtemp(prefix="Sam2Chunks"))   # local disk: much faster than Drive
        WorkDir.mkdir(parents=True, exist_ok=True)
        Yolo = YOLO(DetModel)
        Scale = PreviewWidth / Info["Width"]
        for Number in range(FirstTodo, len(Starts) + 1):
            ChunkStart = Starts[Number - 1]
            Seconds = min(ChunkSeconds, EndSeconds - ChunkStart)
            ChunkStarted = time.time()
            State = _TrackChunk(Yolo, ClipPath, ChunkStart, Seconds, Number, Previous, NextTrackId, Settings,
                                WorkDir, ChunkDir, Scale, Preview)
            Previous, NextTrackId = [(T, B) for T, B in State["Previous"]], State["NextTrackId"]
            Left = (len(Starts) - Number) * (time.time() - Started) / (Number - FirstTodo + 1)
            print(f"  chunk {Number}/{len(Starts)} ({FormatTime(ChunkStart)}): "
                  + (f"{State['Seeds']} seed(s)" if State["Seeds"] else "no confident cow box, skipped")
                  + f", {time.time() - ChunkStarted:.0f} s, ~{Left / 60:.0f} min left")
        del Yolo
        _FreeGpu()
        shutil.rmtree(WorkDir, ignore_errors=True)

    # Rebuild the combined outputs from all chunks.
    Pieces = [pd.read_csv(P) for P in sorted(ChunkDir.glob("Chunk*.csv"))]
    Tracks = pd.concat(Pieces, ignore_index=True) if Pieces else pd.DataFrame(columns=TrackColumns)
    Tracks.to_csv(OutputDir / "Tracks.csv", index=False)
    SummarizeSamTracks(Tracks).to_csv(OutputDir / "TrackSummary.csv", index=False)

    (OutputDir / "PreviewTracks.mp4").unlink(missing_ok=True)
    PreviewPieces = sorted(ChunkDir.glob("Preview*.mp4"))
    if Preview and PreviewPieces:
        ListPath = ChunkDir / "PreviewList.txt"
        ListPath.write_text("".join(f"file '{P.name}'\n" for P in PreviewPieces))
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(ListPath),
                        "-c", "copy", str(OutputDir / "PreviewTracks.mp4")], check=True)
        ListPath.unlink()

    WindowFrames = int((EndSeconds - StartSeconds) * TargetFps)
    Rows = [{"Method": "SAM 2.1 video", **_TrackStats(Tracks, WindowFrames, MinFrames)}]
    if CompareTracksPath and Path(CompareTracksPath).exists():
        Byte = pd.read_csv(CompareTracksPath)
        Byte = Byte[(Byte.TimeSeconds >= StartSeconds) & (Byte.TimeSeconds < EndSeconds)]
        ByteFrames = int((EndSeconds - StartSeconds) * 5)   # 02b ran at 5 fps
        Rows.insert(0, {"Method": "ByteTrack (02b)", **_TrackStats(Byte, ByteFrames, MinFrames * 5 // TargetFps)})
    pd.DataFrame(Rows).to_csv(OutputDir / "Comparison.csv", index=False)

    Stats = {"Chunks": len(Starts), "ChunksRunNow": 0 if FirstTodo is None else len(Starts) - FirstTodo + 1,
             "Seconds": round(time.time() - Started, 1), "Tracks": int(Tracks.TrackId.nunique()), "Boxes": len(Tracks)}
    if len(Tracks):
        Status = np.where(Tracks.YoloIoU >= 0.3, "Agrees", np.where(Tracks.YoloConfidence > 0, "Disagrees", "NoYoloBox"))
        Stats.update({S: round(float((Status == S).mean()), 3) for S in ("Agrees", "NoYoloBox", "Disagrees")})
    print(Stats)
    return Stats


Sam2TrackWindow = Sam2Track   # old name used by the first 02c notebook
