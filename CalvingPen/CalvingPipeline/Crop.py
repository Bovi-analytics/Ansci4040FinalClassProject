"""Step 04: cut each tracked cow out of the clip, with the background masked.

From step 03's ``Tracks.csv`` (SAM 2.1; the ByteTrack baseline's has the same columns):

1. drop duplicate boxes (a weaker box sitting inside a stronger one in the
   same frame, e.g. on the cow's head),
2. keep only tracks that last ``MinFrames`` and average ``MinConfidence``,
3. pick ``CropsPerTrack`` frames per track (or one per ``CropEverySeconds``
   if more), spread over its time span (the most confident box in each
   slice), so the crops show different angles,
4. mask each box with SAM 2.1 Small (Ultralytics ``sam2.1_s.pt``, prompted with
   the box; the pen turns grey, but a calf touching her can stay in) and save
   the crop.

Outputs in ``OutputDir``, overwritten on every run:

- ``KeptTracks.csv``: every track, whether it was kept and why not, and its
  box size relative to the typical track (``RelativeSize``)
- ``Crops.csv``: one row per crop (TrackId, Frame, TimeSeconds, CropPath, box)
- ``Crops/Track<id>/Frame<n>.jpg``: the masked crops
- ``Sheets/Track<id>.jpg``: all crops of one track, for checking by eye
- ``Sheets/Overview.jpg``: one crop per kept track, labelled
"""

import gc
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .Detect import RemoveDuplicateBoxes
from .VideoTools import IterFrames

Grey = 127


def SummarizeKept(Tracks: pd.DataFrame, MinFrames, MinConfidence) -> pd.DataFrame:
    Tracks = Tracks.assign(BoxArea=(Tracks.X2 - Tracks.X1) * (Tracks.Y2 - Tracks.Y1))
    Summary = (Tracks.groupby("TrackId")
               .agg(FirstSeconds=("TimeSeconds", "min"), LastSeconds=("TimeSeconds", "max"),
                    Frames=("Frame", "count"), MeanConfidence=("Confidence", "mean"),
                    MedianBoxArea=("BoxArea", "median"))
               .reset_index())
    Reasons = []
    for Row in Summary.itertuples():
        Why = []
        if Row.Frames < MinFrames:
            Why.append(f"short ({Row.Frames} < {MinFrames} frames)")
        if Row.MeanConfidence < MinConfidence:
            Why.append(f"unsure ({Row.MeanConfidence:.2f} < {MinConfidence})")
        Reasons.append("; ".join(Why))
    Summary["Kept"] = [not R for R in Reasons]
    Summary["DroppedBecause"] = Reasons
    # Box size relative to the typical kept track. Information only: a cow
    # seen end-on or lying is as small as a calf, so it cannot tell them apart.
    Typical = Summary.loc[Summary.Kept, "MedianBoxArea"].median() if Summary.Kept.any() else np.nan
    Summary["RelativeSize"] = (Summary.MedianBoxArea / Typical).round(2)
    return Summary.round({"FirstSeconds": 2, "LastSeconds": 2, "MeanConfidence": 2}).sort_values("FirstSeconds")


def SelectCropFrames(Tracks: pd.DataFrame, CropsPerTrack, CropEverySeconds=None) -> pd.DataFrame:
    """For each track, split its time span into equal slices and take the best
    box in the middle half of each, so two picks are always at least half a
    slice apart (no near-identical crops). Slices = ``CropsPerTrack``, or one
    per ``CropEverySeconds`` if that gives more (SAM tracks can last an hour).
    Best = most confident; for SAM tracks, where YOLO agrees most (``YoloIoU``)."""
    Picked = []
    for _, Track in Tracks.groupby("TrackId"):
        Score = Track.Confidence * Track.YoloIoU if "YoloIoU" in Track else Track.Confidence
        Times = Track.TimeSeconds
        Start, Span = Times.min(), max(Times.max() - Times.min(), 1e-6)
        Slices = CropsPerTrack
        if CropEverySeconds:
            Slices = max(Slices, int(np.ceil(Span / CropEverySeconds)))
        Slices = min(Slices, len(Track))
        Width = Span / Slices
        for I in range(Slices):
            Middle = Times.between(Start + (I + 0.25) * Width, Start + (I + 0.75) * Width)
            Whole = Times.between(Start + I * Width, Start + (I + 1) * Width)
            Candidates = Score[Middle] if Middle.any() else Score[Whole]   # gaps in the track: use the whole slice
            if not Candidates.empty and Candidates.idxmax() not in Picked:
                Picked.append(Candidates.idxmax())
    return Tracks.loc[Picked].reset_index(drop=True)


def _Tile(Image, Size, Label):
    Tile = np.full((Size, Size, 3), 40, np.uint8)
    H, W = Image.shape[:2]
    Scale = Size / max(H, W)
    Small = cv2.resize(Image, (max(1, int(W * Scale)), max(1, int(H * Scale))))
    Y, X = (Size - Small.shape[0]) // 2, (Size - Small.shape[1]) // 2
    Tile[Y:Y + Small.shape[0], X:X + Small.shape[1]] = Small
    cv2.putText(Tile, Label, (4, Size - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3)
    cv2.putText(Tile, Label, (4, Size - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return Tile


def _Sheet(Tiles, Columns, Size):
    Tiles = Tiles + [np.full((Size, Size, 3), 40, np.uint8)] * (-len(Tiles) % Columns)
    return np.vstack([np.hstack(Tiles[I:I + Columns]) for I in range(0, len(Tiles), Columns)])


def CropCows(ClipPath, TracksPath, OutputDir, MinFrames=10, MinConfidence=0.4, CropsPerTrack=24,
             CropEverySeconds=None, MaxSeconds=None, UseMasks=True, SegModel="sam2.1_s.pt", Pad=0.05, TileSize=192) -> dict:
    OutputDir = Path(OutputDir)
    OutputDir.mkdir(parents=True, exist_ok=True)
    for Sub in ("Crops", "Sheets"):          # overwrite: start these folders empty
        shutil.rmtree(OutputDir / Sub, ignore_errors=True)
        (OutputDir / Sub).mkdir()

    Tracks = pd.read_csv(TracksPath)
    if MaxSeconds:
        Tracks = Tracks[Tracks.TimeSeconds <= MaxSeconds]
    Before = len(Tracks)
    Tracks = RemoveDuplicateBoxes(Tracks)
    print(f"{Before - len(Tracks)} duplicate boxes removed of {Before}")

    Summary = SummarizeKept(Tracks, MinFrames, MinConfidence)
    Summary.to_csv(OutputDir / "KeptTracks.csv", index=False)
    KeptIds = Summary.loc[Summary.Kept, "TrackId"]
    print(f"{len(KeptIds)} of {len(Summary)} tracks kept")

    Selected = SelectCropFrames(Tracks[Tracks.TrackId.isin(KeptIds)], CropsPerTrack, CropEverySeconds)
    ByFrame = {Frame: Rows for Frame, Rows in Selected.groupby("Frame")}

    Sam = None
    if UseMasks:
        from ultralytics import SAM
        Sam = SAM(SegModel)

    Rows, Started = [], time.time()
    for Count, (FrameIndex, Frame) in enumerate(IterFrames(ClipPath, ByFrame), 1):
        Boxes = ByFrame[FrameIndex]
        H, W = Frame.shape[:2]
        Masks = None
        if Sam is not None:
            Result = Sam(Frame, bboxes=Boxes[["X1", "Y1", "X2", "Y2"]].values.tolist(), verbose=False)[0]
            Masks = Result.masks.data.cpu().numpy().astype(bool)
            if Masks.shape[1:] != (H, W):
                Masks = np.stack([cv2.resize(M.astype(np.uint8), (W, H)) > 0 for M in Masks])
        for I, Box in enumerate(Boxes.itertuples()):
            PadX, PadY = (Box.X2 - Box.X1) * Pad, (Box.Y2 - Box.Y1) * Pad
            X1, Y1 = int(max(0, Box.X1 - PadX)), int(max(0, Box.Y1 - PadY))
            X2, Y2 = int(min(W, Box.X2 + PadX)), int(min(H, Box.Y2 + PadY))
            Crop = Frame[Y1:Y2, X1:X2].copy()
            MaskFraction = 1.0
            if Masks is not None:
                Mask = Masks[I][Y1:Y2, X1:X2]
                Crop[~Mask] = Grey
                MaskFraction = round(float(Mask.mean()), 3)
            CropPath = Path("Crops") / f"Track{Box.TrackId:04d}" / f"Frame{FrameIndex:07d}.jpg"
            (OutputDir / CropPath).parent.mkdir(exist_ok=True)
            cv2.imwrite(str(OutputDir / CropPath), Crop)
            Rows.append({"TrackId": Box.TrackId, "Frame": FrameIndex, "TimeSeconds": Box.TimeSeconds,
                         "CropPath": CropPath.as_posix(), "X1": X1, "Y1": Y1, "X2": X2, "Y2": Y2,
                         "Confidence": Box.Confidence, "MaskFraction": MaskFraction})
        if Count % 50 == 0:
            print(f"  {Count}/{len(ByFrame)} frames, {Box.TimeSeconds:.0f} s into the clip")

    del Sam
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    Crops = pd.DataFrame(Rows, columns=["TrackId", "Frame", "TimeSeconds", "CropPath", "X1", "Y1", "X2", "Y2",
                                        "Confidence", "MaskFraction"])
    Crops.to_csv(OutputDir / "Crops.csv", index=False)

    Overview = []
    for TrackId, Track in Crops.groupby("TrackId"):
        Tiles = [_Tile(cv2.imread(str(OutputDir / P)), TileSize, f"{T:.0f}s")
                 for P, T in zip(Track.CropPath, Track.TimeSeconds)]
        cv2.imwrite(str(OutputDir / "Sheets" / f"Track{TrackId:04d}.jpg"), _Sheet(Tiles, 8, TileSize))
        Info = Summary.set_index("TrackId").loc[TrackId]
        Label = f"T{TrackId} {Info.FirstSeconds:.0f}-{Info.LastSeconds:.0f}s"
        Middle = Track.iloc[len(Track) // 2]
        Overview.append(_Tile(cv2.imread(str(OutputDir / Middle.CropPath)), TileSize, Label))
    if Overview:
        cv2.imwrite(str(OutputDir / "Sheets" / "Overview.jpg"), _Sheet(Overview, 6, TileSize))

    Stats = {"TracksKept": int(len(KeptIds)), "TracksTotal": int(len(Summary)), "Crops": len(Crops),
             "Seconds": round(time.time() - Started, 1)}
    print(Stats)
    return Stats
