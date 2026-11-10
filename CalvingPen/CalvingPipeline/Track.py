"""Step 03: follow the cow through the clip with SAM 2.1's video mode.

Reads the YOLO boxes of step 02 (``Detections.csv``); runs no detector itself.
SAM 2.1 Small follows each animal's *mask* (outline) using a memory of how
it looked, so it keeps following her while she lies down or strains, where
YOLO loses her. On the reference clip (1080-1980 s) it kept the cow as one
track for 15 minutes with 97 % YOLO agreement; ByteTrack, which links YOLO
boxes by position only, had broken the same stretch into many short tracks
(see ``Baselines/ByteTrack.py``).

How:

1. the clip is cut into chunks of ``ChunkSeconds`` at the frame rate of step
   02 (SAM 2's memory grows with every frame, so an hour cannot go in one go),
2. each chunk starts on its first frame from where SAM left each animal at the
   end of the previous chunk (``CarryOver``); if YOLO has a confident box
   (``SeedConfidence``) on her there, YOLO's box is used instead (corrects
   drift). Confident YOLO boxes elsewhere add new animals (e.g. the calf).
   Without anything to carry over (the first chunk, or she was lost), SAM
   starts at the first frame with a confident YOLO box. Before ``CarryOver``,
   a chunk where YOLO was never sure was skipped (14:00-16:00 of the reference
   clip) and chunks started late,
3. SAM 2.1 Small follows every object through the chunk (``Half=True``: in
   half precision on the GPU, faster),
4. a carried-over animal keeps its TrackId; a new one continues the previous
   chunk's track if its first box overlaps that track's last box (IoU >= ``LinkIou``),
5. every SAM box is compared with YOLO's boxes in the same frame
   (``YoloStatus``): **Agrees** = YOLO sees her in the same place;
   **NoYoloBox** = YOLO saw nothing, SAM kept following her; **Disagrees** =
   YOLO sees a cow elsewhere but not where SAM is (SAM may have drifted).

**Resumable:** every finished chunk is saved in ``OutputDir/Chunks/``. If
the run stops (Colab disconnects, GPU time runs out), run it again with the
same settings and it continues from the first unfinished chunk. Otherwise it
overwrites: a finished run, changed settings, new step 02 boxes or
``Resume=False`` all start over.

Outputs in ``OutputDir`` (rebuilt from the chunks at the end of every run);
``Tracks.csv`` has one row per animal per frame, as step 04 expects:

- ``Tracks.csv``, ``TrackSummary.csv``
- ``PreviewTracks.mp4``: mask (coloured, with outline; the rest darkened),
  box and ID, sped up ``PreviewSpeed`` times (``Preview=False`` skips it)
- ``Comparison.csv``: ByteTrack baseline vs SAM 2 on the same time span (if given)
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

from .Detect import CocoCow, Color, LoadDetections, RemoveDuplicateBoxes
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


def _CutChunk(ClipPath, StartSeconds, Seconds, Fps, OutPath):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{StartSeconds:.3f}", "-i", str(ClipPath),
                    "-t", f"{Seconds:.3f}", "-vf", f"fps={Fps}", "-c:v", "libx264", "-preset", "veryfast",
                    "-crf", "18", "-pix_fmt", "yuv420p", "-an", str(OutPath)], check=True)


def _Encode(SourcePath, OutPath):
    """Same settings for every preview piece, so they can be joined without re-encoding."""
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(SourcePath), "-c:v", "libx264", "-preset", "veryfast",
                    "-crf", "28", "-pix_fmt", "yuv420p", "-an", str(OutPath)], check=True)


def _BoxesByFrame(Detections: pd.DataFrame, Fps) -> dict:
    """{frame number at ``Fps``: (N, 5) array [x1, y1, x2, y2, conf]}."""
    Key = (Detections.TimeSeconds * Fps).round().astype(int)
    return {K: G[["X1", "Y1", "X2", "Y2", "Confidence"]].to_numpy() for K, G in Detections.groupby(Key)}


def _Inside(A, B) -> float:
    """Share of box A that lies inside box B."""
    W = min(A[2], B[2]) - max(A[0], B[0])
    H = min(A[3], B[3]) - max(A[1], B[1])
    return 0.0 if W <= 0 or H <= 0 else W * H / max((A[2] - A[0]) * (A[3] - A[1]), 1e-6)


def _Seeds(Boxes: np.ndarray, SeedConfidence) -> np.ndarray:
    Confident = Boxes[Boxes[:, 4] >= SeedConfidence]
    if len(Confident) < 2:
        return Confident
    Frame = pd.DataFrame(Confident, columns=["X1", "Y1", "X2", "Y2", "Confidence"]).assign(Frame=0)
    return RemoveDuplicateBoxes(Frame)[["X1", "Y1", "X2", "Y2", "Confidence"]].to_numpy()


def DrawMasks(Frame, Objects, Scale, Label):
    """Darken everything, fill each mask with its track colour, outline it."""
    Small = cv2.resize(Frame, None, fx=Scale, fy=Scale)
    Shown = (Small * 0.45).astype(np.uint8)            # background darkened: masked area stands out
    for TrackId, Box, Mask, Status in Objects:
        C = Color(TrackId)
        M = cv2.resize(Mask.astype(np.uint8), (Small.shape[1], Small.shape[0]), interpolation=cv2.INTER_NEAREST)
        Inside = M > 0
        Shown[Inside] = (Small[Inside] * 0.45 + np.array(C) * 0.55).astype(np.uint8)
        Contours, _ = cv2.findContours(M, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(Shown, Contours, -1, C, 2)
        P1, P2 = (int(Box[0] * Scale), int(Box[1] * Scale)), (int(Box[2] * Scale), int(Box[3] * Scale))
        cv2.rectangle(Shown, P1, P2, (255, 255, 255), 1)
        cv2.putText(Shown, f"{TrackId} {Status}", (P1[0], max(P1[1] - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C, 2)
    cv2.putText(Shown, Label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return Shown


def _FreeGpu():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _TrackStats(Tracks: pd.DataFrame, Fps, Seconds, MinSeconds) -> dict:
    """Same measures for any tracker; ``LongTracks`` = tracks of >= ``MinSeconds``."""
    if Tracks.empty:
        return {"Fps": Fps, "Tracks": 0, "LongTracks": 0, "FramesWithCowShare": 0.0,
                "LongestTrackSeconds": 0.0, "MedianTrackSeconds": 0.0}
    Spans = Tracks.groupby("TrackId").TimeSeconds.agg(lambda T: T.max() - T.min())
    return {"Fps": Fps, "Tracks": int(Tracks.TrackId.nunique()), "LongTracks": int((Spans >= MinSeconds).sum()),
            "FramesWithCowShare": round(Tracks.TimeSeconds.round(2).nunique() / max(Seconds * Fps, 1), 3),
            "LongestTrackSeconds": round(float(Spans.max()), 1),
            "MedianTrackSeconds": round(float(Spans.median()), 1)}


def YoloStatus(Tracks: pd.DataFrame) -> np.ndarray:
    return np.where(Tracks.YoloIoU >= 0.3, "Agrees", np.where(Tracks.YoloConfidence > 0, "Disagrees", "NoYoloBox"))


def SummarizeTracks(Tracks: pd.DataFrame) -> pd.DataFrame:
    if Tracks.empty:
        return pd.DataFrame(columns=["TrackId", "FirstSeconds", "LastSeconds", "Frames"])
    Tracks = Tracks.assign(YoloStatus=YoloStatus(Tracks))
    return (Tracks.groupby("TrackId")
            .agg(FirstSeconds=("TimeSeconds", "min"), LastSeconds=("TimeSeconds", "max"),
                 Frames=("Frame", "count"), MedianMaskArea=("MaskArea", "median"),
                 YoloAgrees=("YoloStatus", lambda S: (S == "Agrees").mean()),
                 NoYoloBox=("YoloStatus", lambda S: (S == "NoYoloBox").mean()),
                 YoloDisagrees=("YoloStatus", lambda S: (S == "Disagrees").mean()),
                 MeanConfidence=("Confidence", "mean"))
            .round(3).reset_index().sort_values("FirstSeconds"))


def _TrackChunk(ClipPath, ChunkStart, Seconds, Number, YoloBoxes, Previous, NextTrackId, Settings, WorkDir,
                ChunkDir, Scale, Preview):
    """Track one chunk; writes ``Chunk<n>.csv`` (+ preview piece) and returns the
    state the next chunk needs: last box per track and the next free TrackId."""
    from ultralytics.models.sam import SAM2VideoPredictor

    S, Fps = Settings, Settings["Fps"]
    First, Last = int(round(ChunkStart * Fps)), int(round((ChunkStart + Seconds) * Fps))
    Rows, LastBox, LastKey, PreviewWriter = [], {}, {}, None
    ChunkPath, PreviewTemp = WorkDir / f"Chunk{Number:03d}.mp4", WorkDir / f"Preview{Number:03d}.avi"

    # Seeds: (TrackId or None, box). Carry over animals SAM still had at the end of the last chunk.
    Carried = [(T, B) for T, B, Key in Previous if S["CarryOver"] and Key is not None and Key >= First - 2 * Fps]
    Confident = _Seeds(YoloBoxes.get(First, np.zeros((0, 5))), S["SeedConfidence"])
    Seeds, Used, Sources = [], set(), []
    for TrackId, Box in Carried:
        Ious = [BoxIou(Box, Y) for Y in Confident]
        Best = int(np.argmax(Ious)) if Ious else None
        if Best is not None and Ious[Best] >= 0.3 and Best not in Used:
            Seeds.append((TrackId, Confident[Best, :4])); Used.add(Best); Sources.append("yolo")
        else:
            Seeds.append((TrackId, np.asarray(Box))); Sources.append("carried")
    for I in range(len(Confident)):   # new animals: confident YOLO boxes not already on a carried one
        if I not in Used and all(_Inside(Confident[I], B) < 0.8 for _, B in Seeds):
            Seeds.append((None, Confident[I, :4]))
    Sources += ["new"] * (len(Seeds) - len(Sources))
    SeedKey = First if Seeds else next((K for K in range(First, Last) if K in YoloBoxes
                                        and (YoloBoxes[K][:, 4] >= S["SeedConfidence"]).any()), None)
    if Seeds == [] and SeedKey is not None:
        Seeds = [(None, B[:4]) for B in _Seeds(YoloBoxes[SeedKey], S["SeedConfidence"])]
        Sources = ["new"] * len(Seeds)

    if SeedKey is not None:
        SeedStart = SeedKey / Fps            # SAM 2 takes its prompts on the first frame: start the chunk there
        _CutChunk(ClipPath, SeedStart, ChunkStart + Seconds - SeedStart, Fps, ChunkPath)
        Predictor = SAM2VideoPredictor(overrides=dict(conf=0.25, task="segment", mode="predict", imgsz=S["ImageSize"],
                                                      model=S["SegModel"], half=S["Half"], save=False, verbose=False))
        Ids = {I: T for I, (T, _) in enumerate(Seeds) if T is not None}   # SAM object index -> TrackId
        Boxes = [[float(V) for V in B] for _, B in Seeds]
        for I, R in enumerate(Predictor(source=str(ChunkPath), bboxes=Boxes, stream=True)):
            TimeSeconds = round((SeedKey + I) / Fps, 3)
            FrameYolo = YoloBoxes.get(SeedKey + I, np.zeros((0, 5)))
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
                    LastKey[Ids[Obj]] = SeedKey + I
                    Objects.append((Ids[Obj], Box, Mask, Status))
            if Preview:
                Drawn = DrawMasks(R.orig_img, Objects, Scale, f"{TimeSeconds:7.1f}s  chunk {Number}")
                if PreviewWriter is None:
                    PreviewWriter = cv2.VideoWriter(str(PreviewTemp), cv2.VideoWriter_fourcc(*"MJPG"),
                                                    Fps * S["PreviewSpeed"], (Drawn.shape[1], Drawn.shape[0]))
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
             "SeedSources": Sources, "StartedAt": None if SeedKey is None else round(SeedKey / Fps, 3),
             "Previous": [[T, B, LastKey[T]] for T, B in LastBox.items()], "NextTrackId": NextTrackId}
    # Written last: a chunk only counts as done once its .json exists.
    (ChunkDir / f"Chunk{Number:03d}.json").write_text(json.dumps(State))
    return State


def TrackCows(ClipPath, DetectionsDir, OutputDir, StartSeconds=None, EndSeconds=None, ChunkSeconds=120,
              SeedConfidence=0.4, SegModel="sam2.1_s.pt", ImageSize=1024, LinkIou=0.3, CompareTracksPath=None,
              CarryOver=True, Half=False, Preview=True, PreviewSpeed=4, PreviewWidth=960, Resume=True,
              WorkDir=None) -> dict:
    """Track the span step 02 covered (or ``StartSeconds``..``EndSeconds`` inside it)."""
    OutputDir, ChunkDir = Path(OutputDir), Path(OutputDir) / "Chunks"
    OutputDir.mkdir(parents=True, exist_ok=True)
    Detections, Detect = LoadDetections(DetectionsDir)
    Info = ProbeVideo(ClipPath)
    Fps = Detect["TargetFps"]
    StartSeconds = Detect["StartSeconds"] if StartSeconds is None else max(StartSeconds, Detect["StartSeconds"])
    EndSeconds = min(EndSeconds or Detect["EndSeconds"], Detect["EndSeconds"])
    import hashlib
    DetectionsHash = hashlib.md5((Path(DetectionsDir) / "Detections.csv").read_bytes()).hexdigest()
    Settings = {"Clip": str(ClipPath), "Detections": str(DetectionsDir), "DetectionsHash": DetectionsHash,
                "StartSeconds": StartSeconds,
                "EndSeconds": EndSeconds, "Fps": Fps, "ChunkSeconds": ChunkSeconds, "SeedConfidence": SeedConfidence,
                "SegModel": SegModel, "ImageSize": ImageSize, "LinkIou": LinkIou, "CarryOver": CarryOver,
                "Half": Half, "Preview": Preview,
                "PreviewSpeed": PreviewSpeed, "PreviewWidth": PreviewWidth, "ClipFps": Info["Fps"]}

    # Overwrite rule: start over unless the last run with exactly these settings
    # (and the same step 02 boxes) stopped part-way; only then continue it.
    SettingsPath = ChunkDir / "Settings.json"
    Same = SettingsPath.exists() and json.loads(SettingsPath.read_text()) == Settings
    Total = len(np.arange(StartSeconds, EndSeconds - 1e-6, ChunkSeconds))
    Finished = Same and all((ChunkDir / f"Chunk{N:03d}.json").exists() for N in range(1, Total + 1))
    if not (Resume and Same) or Finished:
        if ChunkDir.exists():
            print("Starting over: " + ("Resume=False" if not Resume else
                                       "the last run finished" if Finished else "settings or step 02 boxes changed"))
        shutil.rmtree(ChunkDir, ignore_errors=True)
    ChunkDir.mkdir(exist_ok=True)
    SettingsPath.write_text(json.dumps(Settings, indent=1))

    Starts = [round(float(S), 3) for S in np.arange(StartSeconds, EndSeconds - 1e-6, ChunkSeconds)]
    Done = {N for N in range(1, len(Starts) + 1) if (ChunkDir / f"Chunk{N:03d}.json").exists()}
    FirstTodo = next((N for N in range(1, len(Starts) + 1) if N not in Done), None)
    print(f"{Info['Video']}: {FormatTime(StartSeconds)}-{FormatTime(EndSeconds)} of the clip, {Fps} fps, "
          f"{len(Starts)} chunks of {ChunkSeconds} s; {len(Done)} already done")

    Started = time.time()
    if FirstTodo is not None:
        if FirstTodo > 1:
            State = json.loads((ChunkDir / f"Chunk{FirstTodo - 1:03d}.json").read_text())
            Previous, NextTrackId = [tuple(P) + (None,) * (3 - len(P)) for P in State["Previous"]], State["NextTrackId"]
        else:
            Previous, NextTrackId = [], 1
        YoloBoxes = _BoxesByFrame(Detections, Fps)
        WorkDir = Path(WorkDir or tempfile.mkdtemp(prefix="Sam2Chunks"))   # local disk: much faster than Drive
        WorkDir.mkdir(parents=True, exist_ok=True)
        Scale = PreviewWidth / Info["Width"]
        for Number in range(FirstTodo, len(Starts) + 1):
            ChunkStart = Starts[Number - 1]
            Seconds = min(ChunkSeconds, EndSeconds - ChunkStart)
            ChunkStarted = time.time()
            State = _TrackChunk(ClipPath, ChunkStart, Seconds, Number, YoloBoxes, Previous, NextTrackId, Settings,
                                WorkDir, ChunkDir, Scale, Preview)
            Previous, NextTrackId = [tuple(P) for P in State["Previous"]], State["NextTrackId"]
            Left = (len(Starts) - Number) * (time.time() - Started) / (Number - FirstTodo + 1)
            Kinds = ", ".join(f"{State['SeedSources'].count(K)} {K}" for K in ("carried", "yolo", "new")
                              if State["SeedSources"].count(K))
            Late = (f", started {State['StartedAt'] - ChunkStart:.0f} s in"
                    if State["StartedAt"] is not None and State["StartedAt"] > ChunkStart + 1 else "")
            print(f"  chunk {Number}/{len(Starts)} ({FormatTime(ChunkStart)}): "
                  + (f"{State['Seeds']} seed(s) ({Kinds}){Late}" if State["Seeds"] else "no cow to follow, skipped")
                  + f", {time.time() - ChunkStarted:.0f} s, ~{Left / 60:.0f} min left")
        shutil.rmtree(WorkDir, ignore_errors=True)

    # Rebuild the combined outputs from all chunks.
    Pieces = [Piece for P in sorted(ChunkDir.glob("Chunk*.csv")) if len(Piece := pd.read_csv(P))]
    Tracks = pd.concat(Pieces, ignore_index=True) if Pieces else pd.DataFrame(columns=TrackColumns)
    Tracks.to_csv(OutputDir / "Tracks.csv", index=False)
    SummarizeTracks(Tracks).to_csv(OutputDir / "TrackSummary.csv", index=False)

    (OutputDir / "PreviewTracks.mp4").unlink(missing_ok=True)
    PreviewPieces = sorted(ChunkDir.glob("Preview*.mp4"))
    if Preview and PreviewPieces:
        ListPath = ChunkDir / "PreviewList.txt"
        ListPath.write_text("".join(f"file '{P.name}'\n" for P in PreviewPieces))
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(ListPath),
                        "-c", "copy", str(OutputDir / "PreviewTracks.mp4")], check=True)
        ListPath.unlink()

    Span = EndSeconds - StartSeconds
    Rows = [{"Method": "SAM 2.1 video (step 03)", **_TrackStats(Tracks, Fps, Span, 5)}]
    if CompareTracksPath and Path(CompareTracksPath).exists():
        Byte = pd.read_csv(CompareTracksPath)
        Byte = Byte[(Byte.TimeSeconds >= StartSeconds) & (Byte.TimeSeconds < EndSeconds)]
        Rows.insert(0, {"Method": "ByteTrack (baseline)", **_TrackStats(Byte, 5, Span, 5)})   # baseline ran at 5 fps
    pd.DataFrame(Rows).to_csv(OutputDir / "Comparison.csv", index=False)

    Stats = {"Chunks": len(Starts), "ChunksRunNow": 0 if FirstTodo is None else len(Starts) - FirstTodo + 1,
             "Seconds": round(time.time() - Started, 1), "Tracks": int(Tracks.TrackId.nunique()), "Boxes": len(Tracks)}
    if len(Tracks):
        Status = YoloStatus(Tracks)
        Stats.update({S: round(float((Status == S).mean()), 3) for S in ("Agrees", "NoYoloBox", "Disagrees")})
    print(Stats)
    return Stats
