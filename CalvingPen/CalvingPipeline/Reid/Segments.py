"""Stages 1-4 of the re-ID design in one pass over a video: read frames at
``TargetFps`` (``Frames.SampleFrames``: times from the AVI chunks, never the
broken container timestamps), box the cows (YOLO), follow them (ByteTrack), and
every ``EverySeconds`` mask every cow in that frame (SAM 2.1 image mode,
prompted with the YOLO boxes) and save its canonical crop.

The tracker only follows; it never names. A TrackId is an anonymous segment,
and one cow can have hundreds of them: the identity layer names each one.

Outputs in ``OutputDir`` (overwritten when the settings change; a stopped run
continues where it stopped, with new TrackIds after the old ones):

- ``Tracks.csv``: one row per box per sampled frame (Tick, TimeSeconds, TrackId, box, Confidence)
- ``Crops.csv``: one row per canonical crop, with the gate numbers
  (MaskArea, NeighbourIos, Elongation, BoxAspect, ColouredFraction, Night)
- ``Crops/Track<id>/T<seconds>.jpg``: the 224 x 224 canonical crops
- ``FirstFrame.jpg``, ``LastFrame.jpg``: for the seam check between two parts
- ``Run.json``: settings, how far it got, where the recording really ends
"""

import csv
import gc
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .Canonical import CanonicalCrop, ColouredFraction, Elongation
from .Frames import LocalCopy, SampleFrames

CocoCow = 19
TrackColumns = ["Tick", "TimeSeconds", "TrackId", "X1", "Y1", "X2", "Y2", "Confidence"]
CropColumns = ["TrackId", "Tick", "TimeSeconds", "CropPath", "X1", "Y1", "X2", "Y2", "Confidence", "MaskArea",
               "NeighbourIos", "Elongation", "BoxAspect", "ColouredFraction", "FrameSaturation", "Night"]


def _TrackerConfig(BufferFrames: int, Folder: Path) -> str:
    import ultralytics

    Source = Path(ultralytics.__file__).parent / "cfg" / "trackers" / "bytetrack.yaml"
    Text = "\n".join(f"track_buffer: {BufferFrames}" if L.startswith("track_buffer") else L
                     for L in Source.read_text().splitlines())
    Path_ = Folder / "ByteTrack.yaml"
    Path_.write_text(Text)
    return str(Path_)


def _FrameSaturation(Frame) -> float:
    Small = cv2.resize(Frame, (160, 90), interpolation=cv2.INTER_AREA)
    return float(cv2.cvtColor(Small, cv2.COLOR_BGR2HSV)[..., 1].mean())


def _Masks(Sam, Frame, Boxes):
    H, W = Frame.shape[:2]
    Result = Sam(Frame, bboxes=[list(B) for B in Boxes], verbose=False)[0]
    Masks = Result.masks.data.cpu().numpy().astype(bool)
    if Masks.shape[1:] != (H, W):
        Masks = np.stack([cv2.resize(M.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0 for M in Masks])
    return Masks


def _NeighbourIos(Masks):
    """Per mask: the largest share of its pixels that another mask also covers."""
    Areas = Masks.reshape(len(Masks), -1).sum(1)
    Out = np.zeros(len(Masks))
    for I in range(len(Masks)):
        for J in range(len(Masks)):
            if I != J and Areas[I]:
                Out[I] = max(Out[I], float((Masks[I] & Masks[J]).sum()) / Areas[I])
    return Out


def TrackAndCrop(VideoPath, OutputDir, TargetFps=5, EverySeconds=15, StartSeconds=0.0, EndSeconds=None,
                 DetModel="yolo26n.pt", ImageSize=1024, Confidence=0.25, TrackBufferSeconds=10,
                 SegModel="sam2.1_s.pt", NativeFps=None, NightSaturation=15, FlushSeconds=60, Redo=False) -> dict:
    from ultralytics import SAM, YOLO

    OutputDir = Path(OutputDir)
    OutputDir.mkdir(parents=True, exist_ok=True)
    Settings = {"Video": Path(VideoPath).name, "TargetFps": TargetFps, "EverySeconds": EverySeconds,
                "StartSeconds": StartSeconds, "EndSeconds": EndSeconds, "DetModel": DetModel, "ImageSize": ImageSize,
                "Confidence": Confidence, "TrackBufferSeconds": TrackBufferSeconds, "SegModel": SegModel,
                "NativeFps": NativeFps}
    RunPath = OutputDir / "Run.json"
    Run = json.loads(RunPath.read_text()) if RunPath.exists() else {}
    Same = Run.get("Settings") == Settings and not Redo
    if Same and Run.get("Finished"):
        print(f"{Settings['Video']}: already done ({Run['Tracks']} tracks, {Run['Crops']} crops); Redo=True to redo")
        return Run
    Resume = Same and (OutputDir / "Tracks.csv").exists() and Run.get("DoneSeconds") is not None
    if not Resume:
        for Name in ("Tracks.csv", "Crops.csv", "FirstFrame.jpg", "LastFrame.jpg"):
            (OutputDir / Name).unlink(missing_ok=True)
        Run = {"Settings": Settings, "Finished": False, "DoneSeconds": None, "TrackIdOffset": 0}
    else:   # keep only what was flushed before the stop; the new tracker numbers its tracks after the old ones
        import pandas as pd
        Done = Run["DoneSeconds"]
        for Name in ("Tracks.csv", "Crops.csv"):
            if (OutputDir / Name).exists():
                Table = pd.read_csv(OutputDir / Name)
                Table[Table.TimeSeconds <= Done].to_csv(OutputDir / Name, index=False)
        Tracks = pd.read_csv(OutputDir / "Tracks.csv")
        Run["TrackIdOffset"] = int(Tracks.TrackId.max()) if len(Tracks) else Run.get("TrackIdOffset", 0)
        print(f"Continuing from {Done:.0f} s (new TrackIds after {Run['TrackIdOffset']})")

    Local = LocalCopy(VideoPath)
    From = StartSeconds if not Resume else Run["DoneSeconds"] + 1.0 / TargetFps
    Model, Sam = YOLO(DetModel), SAM(SegModel)
    Tracker = _TrackerConfig(int(TrackBufferSeconds * TargetFps), OutputDir)
    CropEvery = max(1, int(round(EverySeconds * TargetFps)))
    Offset, Info = Run["TrackIdOffset"], {}
    NewFiles = not (OutputDir / "Tracks.csv").exists()
    TrackFile = (OutputDir / "Tracks.csv").open("a", newline="")
    CropFile = (OutputDir / "Crops.csv").open("a", newline="")
    TrackWriter, CropWriter = csv.writer(TrackFile), csv.writer(CropFile)
    if NewFiles:
        TrackWriter.writerow(TrackColumns)
        CropWriter.writerow(CropColumns)
    Started, Count, Crops, LastFlush, LastFrame, T = time.time(), 0, 0, From, None, From
    NextPrint = From + 600
    print(f"{Settings['Video']}: {TargetFps} fps from {From:.0f} s, a crop of every cow every {EverySeconds} s")
    try:
        for Tick, T, Frame in SampleFrames(Local, TargetFps, From, EndSeconds, NativeFps, Info):
            if Count == 0 and not Resume:
                cv2.imwrite(str(OutputDir / "FirstFrame.jpg"), Frame)
            Result = Model.track(Frame, persist=True, tracker=Tracker, classes=[CocoCow], conf=Confidence,
                                 imgsz=ImageSize, verbose=False)[0]
            Boxes = []
            if Result.boxes is not None and Result.boxes.id is not None:
                for Box, Id, Conf in zip(Result.boxes.xyxy.cpu().numpy().round(1), Result.boxes.id.int().tolist(),
                                         Result.boxes.conf.tolist()):
                    Boxes.append((Id + Offset, *Box.tolist(), round(Conf, 3)))
                    TrackWriter.writerow([Tick, T, Id + Offset, *Box.tolist(), round(Conf, 3)])
            if Boxes and Count % CropEvery == 0:
                Masks = _Masks(Sam, Frame, [B[1:5] for B in Boxes])
                Overlap = _NeighbourIos(Masks)
                Saturation = _FrameSaturation(Frame)
                Pil = Image.fromarray(cv2.cvtColor(Frame, cv2.COLOR_BGR2RGB))
                for (Id, X1, Y1, X2, Y2, Conf), Mask, Ios in zip(Boxes, Masks, Overlap):
                    Crop, CropMask = CanonicalCrop(Pil, Mask, ReturnMask=True)
                    if Crop is None:
                        continue
                    CropPath = Path("Crops") / f"Track{Id:05d}" / f"T{int(round(T)):06d}.jpg"
                    (OutputDir / CropPath).parent.mkdir(parents=True, exist_ok=True)
                    Crop.save(OutputDir / CropPath, quality=88)
                    CropWriter.writerow([Id, Tick, T, CropPath.as_posix(), X1, Y1, X2, Y2, Conf, int(Mask.sum()),
                                         round(float(Ios), 4), round(Elongation(Mask), 3),
                                         round((X2 - X1) / max(Y2 - Y1, 1), 3), round(ColouredFraction(Crop, CropMask), 4),
                                         round(Saturation, 1), Saturation < NightSaturation])
                    Crops += 1
            Count, LastFrame = Count + 1, Frame
            if T - LastFlush >= FlushSeconds:
                TrackFile.flush(); CropFile.flush()
                Run["DoneSeconds"], LastFlush = T, T
                RunPath.write_text(json.dumps(Run, indent=1))
                if T >= NextPrint:
                    NextPrint = T + 600
                    print(f"  {T / 60:.0f} min of video done, {Count / (time.time() - Started):.1f} frames/s, {Crops} crops")
    finally:
        TrackFile.close(); CropFile.close()
        del Model, Sam
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    import pandas as pd
    if LastFrame is not None:
        cv2.imwrite(str(OutputDir / "LastFrame.jpg"), LastFrame)
    Tracks, CropTable = pd.read_csv(OutputDir / "Tracks.csv"), pd.read_csv(OutputDir / "Crops.csv")
    Run.update({"Finished": True, "DoneSeconds": T, "LastRealSeconds": Info.get("LastRealSeconds"),
                "RealPictures": Info.get("RealPictures"), "NativeFps": Info.get("NativeFps"),
                "Tracks": int(Tracks.TrackId.nunique()), "Crops": int(len(CropTable)),
                "Minutes": round((time.time() - Started) / 60, 1)})
    RunPath.write_text(json.dumps(Run, indent=1))
    print({K: Run[K] for K in ("Tracks", "Crops", "LastRealSeconds", "RealPictures", "Minutes")})
    return Run


def SeamCheck(Part1Dir, Part2Dir, OutPath=None) -> dict:
    """Does part 2 start where part 1 ends? Compares part 1's last frame with
    part 2's first (mean grey difference, 0-255) and saves them side by side."""
    A = cv2.imread(str(Path(Part1Dir) / "LastFrame.jpg"))
    B = cv2.imread(str(Path(Part2Dir) / "FirstFrame.jpg"))
    if A is None or B is None:
        raise FileNotFoundError("Run TrackAndCrop on both parts first (LastFrame.jpg / FirstFrame.jpg)")
    Size = (480, 270)
    Ga, Gb = [cv2.cvtColor(cv2.resize(X, Size), cv2.COLOR_BGR2GRAY).astype(np.float32) for X in (A, B)]
    Difference = float(np.abs(Ga - Gb).mean())
    if OutPath:
        cv2.imwrite(str(OutPath), np.hstack([cv2.resize(A, Size), cv2.resize(B, Size)]))
    Verdict = ("same scene: part 2 probably follows part 1" if Difference < 12 else
               "different picture: check by eye (a cow may have moved, or the parts are not consecutive)")
    return {"MeanGreyDifference": round(Difference, 1), "Verdict": Verdict}
