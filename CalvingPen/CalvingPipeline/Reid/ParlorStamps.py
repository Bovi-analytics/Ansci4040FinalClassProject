"""Named parlor crops from the parlor group's hand-timed cow list.

The parlor group (``ParlorIDMatch`` on the Shared Drive) wrote down, for one
milking, the second each cow walks into one side of the parlor
(``TimeSecond.xlsx``: seconds in the parlor video file). No detector is
needed: YOLO cannot find the cows in this fisheye view, but a person can.

1. ``ReadStamps``: the times, split into turns where the gap is > 2 min
   (one turn = one side filling up, 28 cows).
2. ``LabelStamps``: turn *i* of the list = the *i*-th turn of that side and
   milking in the BoviSync parlor report; the *k*-th cow in = stall *k*
   (``FirstCowStall`` = 1: the first cow walks to stall 1, the parlor
   group's rule; the right side of the picture = stalls 1-28). The report
   gives the cow in that stall; cows without a stall in the report stay
   unnamed.
3. ``StampCheck``: pictures around the first times, with a grid, to choose
   ``Point``: a spot every cow walks over a moment after her time.
4. ``CropStamps``: at each time + ``Offsets`` seconds, SAM 2.1 cuts out the
   cow under ``Point`` and the canonical crop is saved. The folder looks like
   a ``TrackAndCrop`` folder (Crops.csv, Tracks.csv, FirstFrame.jpg; one
   "track" per cow), so ``EmbedBands`` and ``LoadVideo`` work on it, and
   ``Stamps.csv`` gives each track's CowId.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .Canonical import CanonicalCrop, ColouredFraction, Elongation
from .Frames import LocalCopy, SampleFrames
from .Segments import CropColumns, TrackColumns, _FrameSaturation


def ReadStamps(Path_, GapSeconds=120) -> pd.DataFrame:
    """Times (s in the video) from the first column named TimeSecond of an
    .xlsx/.csv; adds Turn (1, 2, ...) and Order (1.. within the turn)."""
    Path_ = Path(Path_)
    Table = pd.read_excel(Path_) if Path_.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(Path_)
    Times = pd.to_numeric(Table["TimeSecond"], errors="coerce").dropna().astype(float).sort_values().to_numpy()
    Turn = np.r_[0, np.cumsum(np.diff(Times) > GapSeconds)] + 1
    Out = pd.DataFrame({"TimeSeconds": Times, "Turn": Turn})
    Out["Order"] = Out.groupby("Turn").cumcount() + 1
    Out.insert(0, "StampId", np.arange(1, len(Out) + 1))
    return Out


def LabelStamps(Stamps, Report, Session=1, Side="A", FirstCowStall=1, StallsPerSide=28) -> pd.DataFrame:
    """CowId per stamp from the report (see the module notes); prints how
    the turns line up (the list's turn should start a few minutes before the
    report's attach time)."""
    Turns = (Report[(Report.Session == Session) & (Report.Side == Side)]
             .groupby("Turn").TurnMinute.first().sort_values())
    Base = 0 if Side == "A" else StallsPerSide
    Rows = []
    for (N, G), (TurnName, Minute) in zip(Stamps.groupby("Turn"), Turns.items()):
        InTurn = Report[Report.Turn == TurnName].set_index("Stall").CowId
        for S in G.itertuples():
            K = S.Order if FirstCowStall == 1 else StallsPerSide + 1 - S.Order
            Rows.append({"StampId": S.StampId, "Turn": TurnName, "Stall": Base + K,
                         "CowId": InTurn.get(Base + K)})
        print(f"list turn {N}: {len(G)} cows from {G.TimeSeconds.min() / 3600:.2f} h  ->  report {TurnName}, attach "
              f"{int(Minute // 60)}:{int(Minute % 60):02d}, {InTurn.size} cows with a stall")
    Out = Stamps.rename(columns={"Turn": "ListTurn"}).merge(pd.DataFrame(Rows), on="StampId", how="left")
    Out["CowId"] = Out.CowId.astype("Int64")
    return Out


def _Frames(VideoPath, Times, NativeFps=None, Step=0.5):
    """{wanted time: frame}: one pass over the span, a frame every ``Step`` s."""
    Wanted = sorted(set(Times))
    Got, I = {}, 0
    for _, T, Frame in SampleFrames(VideoPath, 1 / Step, max(0.0, Wanted[0] - Step), Wanted[-1] + Step, NativeFps):
        while I < len(Wanted) and Wanted[I] <= T + Step / 2:
            if abs(Wanted[I] - T) <= Step / 2:
                Got[Wanted[I]] = Frame.copy()
            I += 1
    return Got


def StampCheck(VideoPath, Stamps, OutPath, Count=3, Offsets=(-2, 0, 2, 4), Width=480, NativeFps=None):
    """Pictures at the first ``Count`` times + ``Offsets`` (rows = cows,
    columns = offsets), with a 20-px grid and coordinates, to choose ``Point``
    (in the video's own pixels)."""
    Local = LocalCopy(VideoPath)
    Rows = Stamps.head(Count)
    Want = [round(T + O, 1) for T in Rows.TimeSeconds for O in Offsets]
    Frames = _Frames(Local, Want, NativeFps)
    Tiles = []
    for T in Want:
        F = Frames.get(T)
        if F is None:
            F = np.zeros((240, 352, 3), np.uint8)
        F = F.copy()
        H, W = F.shape[:2]
        for X in range(0, W, 20):
            cv2.line(F, (X, 0), (X, H), (0, 255, 255) if X % 100 == 0 else (90, 90, 90), 1)
        for Y in range(0, H, 20):
            cv2.line(F, (0, Y), (W, Y), (0, 255, 255) if Y % 100 == 0 else (90, 90, 90), 1)
        for X in range(0, W, 100):
            for Y in range(0, H, 100):
                cv2.putText(F, f"{X},{Y}", (X + 2, Y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
        Scale = Width / W
        F = cv2.resize(F, None, fx=Scale, fy=Scale, interpolation=cv2.INTER_NEAREST)
        cv2.putText(F, f"{T:.1f} s", (4, F.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        Tiles.append(F)
    Grid = [np.hstack(Tiles[I:I + len(Offsets)]) for I in range(0, len(Tiles), len(Offsets))]
    cv2.imwrite(str(OutPath), np.vstack(Grid))
    return OutPath


def _MotionBox(Frame, Before, Region, MinChange=8.0, Share=0.3, Grow=0.15):
    """Box (x1, y1, x2, y2) around the moving area inside ``Region`` that
    holds the largest change between ``Before`` and ``Frame`` (the cow
    walking in): the pixels changing at least ``Share`` of that peak,
    connected to it, grown by ``Grow``. None if nothing moved."""
    A = cv2.GaussianBlur(cv2.cvtColor(Frame, cv2.COLOR_BGR2GRAY), (9, 9), 0).astype(np.float32)
    B = cv2.GaussianBlur(cv2.cvtColor(Before, cv2.COLOR_BGR2GRAY), (9, 9), 0).astype(np.float32)
    Change = cv2.GaussianBlur(np.abs(A - B), (21, 21), 0)
    X1, Y1, X2, Y2 = Region
    Inside = Change[Y1:Y2, X1:X2]
    if Inside.size == 0 or Inside.max() < MinChange:
        return None
    Y, X = np.unravel_index(int(np.argmax(Inside)), Inside.shape)
    _, Labels = cv2.connectedComponents((Inside >= Share * Inside.max()).astype(np.uint8))
    Ys, Xs = np.nonzero(Labels == Labels[Y, X])
    BX1, BY1, BX2, BY2 = Xs.min() + X1, Ys.min() + Y1, Xs.max() + X1, Ys.max() + Y1
    GW, GH = (BX2 - BX1) * Grow, (BY2 - BY1) * Grow
    H, W = Frame.shape[:2]
    return [int(max(0, BX1 - GW)), int(max(0, BY1 - GH)), int(min(W - 1, BX2 + GW)), int(min(H - 1, BY2 + GH))]


def CropStamps(VideoPath, Stamps, OutputDir, Point, Offsets=(1.0, 2.0, 3.0), SegModel="sam2.1_s.pt", NativeFps=None,
               MinArea=1000, Region=None, Redo=False) -> pd.DataFrame:
    """Per stamp and offset: SAM's mask of the cow under ``Point`` (x, y in
    the video's pixels) and its canonical crop. ``Stamps`` from
    ``LabelStamps``; ``TrackId`` = ``StampId``. ``Point = "Motion"``: at each
    time, the spot inside ``Region`` (default: the right third of the
    picture, the stalls 1-28 side, away from the pit) where the picture
    changed most in the last 0.5 s, i.e. the cow walking in; SAM gets the box
    around that moving area. ``MinArea``: smaller masks (a patch, not a whole
    cow; ~2,000-5,000 px in the 352 x 240 parlor video) are dropped. Writes Crops.csv, Tracks.csv,
    FirstFrame.jpg, Stamps.csv and Run.json; skipped when done with the same
    settings."""
    from PIL import Image
    from ultralytics import SAM

    OutputDir = Path(OutputDir)
    (OutputDir / "Crops").mkdir(parents=True, exist_ok=True)
    Motion = isinstance(Point, str)
    Settings = {"Video": Path(VideoPath).name, "Point": Point if Motion else list(map(int, Point)),
                "Region": None if Region is None else list(map(int, Region)), "Offsets": list(Offsets),
                "Stamps": int(len(Stamps)), "SegModel": SegModel}
    RunPath = OutputDir / "Run.json"
    if RunPath.exists() and not Redo and json.loads(RunPath.read_text()).get("Settings") == Settings:
        print(f"{OutputDir.name}: already done; Redo=True to redo")
        return pd.read_csv(OutputDir / "Crops.csv")
    Local = LocalCopy(VideoPath)
    Want = {}
    for S in Stamps.itertuples():
        for O in Offsets:
            Want.setdefault(round(S.TimeSeconds + O, 1), []).append(S.StampId)
    Frames = _Frames(Local, list(Want) + ([round(T - 0.5, 1) for T in Want] if Motion else []), NativeFps)
    Sam = SAM(SegModel)
    Crops, Tracks, First = [], [], None
    for T in sorted(Want):
        Frame = Frames.get(T)
        if Frame is None:
            continue
        First = Frame if First is None else First
        if Motion:
            H, W = Frame.shape[:2]
            Before = Frames.get(round(T - 0.5, 1))
            Box = None if Before is None else _MotionBox(Frame, Before, Region or (int(W * 2 / 3), 0, W, H))
            if Box is None:
                continue
            Result = Sam(Frame, bboxes=[Box], verbose=False)[0]
        else:
            Result = Sam(Frame, points=[list(map(int, Point))], labels=[1], verbose=False)[0]
        if Result.masks is None or not len(Result.masks.data):
            continue
        Masks = Result.masks.data.cpu().numpy().astype(bool)
        Mask = Masks[int(np.argmax(Masks.reshape(len(Masks), -1).sum(1)))]
        if Mask.shape != Frame.shape[:2]:
            Mask = cv2.resize(Mask.astype(np.uint8), Frame.shape[1::-1], interpolation=cv2.INTER_NEAREST) > 0
        if Mask.sum() < MinArea:
            continue
        Ys, Xs = np.nonzero(Mask)
        Box = [float(Xs.min()), float(Ys.min()), float(Xs.max()), float(Ys.max())]
        Pil = Image.fromarray(cv2.cvtColor(Frame, cv2.COLOR_BGR2RGB))
        Crop, CropMask = CanonicalCrop(Pil, Mask, ReturnMask=True, MinPixels=MinArea)
        if Crop is None:
            continue
        Saturation = _FrameSaturation(Frame)
        for StampId in Want[T]:
            CropPath = Path("Crops") / f"Track{StampId:05d}" / f"T{int(round(T * 10)):07d}.jpg"
            (OutputDir / CropPath).parent.mkdir(parents=True, exist_ok=True)
            Crop.save(OutputDir / CropPath, quality=90)
            Crops.append([StampId, 0, T, CropPath.as_posix(), *Box, 1.0, int(Mask.sum()), 0.0, round(Elongation(Mask), 3),
                          round((Box[2] - Box[0]) / max(Box[3] - Box[1], 1), 3), round(ColouredFraction(Crop, CropMask), 4),
                          round(Saturation, 1), False])
            Tracks.append([0, T, StampId, *Box, 1.0])
    Table = pd.DataFrame(Crops, columns=CropColumns)
    Table.to_csv(OutputDir / "Crops.csv", index=False)
    pd.DataFrame(Tracks, columns=TrackColumns).to_csv(OutputDir / "Tracks.csv", index=False)
    Stamps.to_csv(OutputDir / "Stamps.csv", index=False)
    if First is not None:
        cv2.imwrite(str(OutputDir / "FirstFrame.jpg"), First)
    RunPath.write_text(json.dumps({"Settings": Settings, "Finished": True, "Crops": len(Table),
                                   "Cows": int(Table.TrackId.nunique()) if len(Table) else 0}, indent=1))
    print(f"{OutputDir.name}: {len(Table)} crops of {Table.TrackId.nunique() if len(Table) else 0} of {len(Stamps)} cows")
    return Table


def StampSheet(OutputDir, PerCow=3, Cows=40, TileSize=110) -> Path:
    """``StampSheet.jpg``: the crops of the first ``Cows`` stamps, a row per
    cow labelled with her stall and CowId, to check the crops show one cow."""
    from .Labels import _Sheet, _Tile

    OutputDir = Path(OutputDir)
    Crops = pd.read_csv(OutputDir / "Crops.csv")
    Stamps = pd.read_csv(OutputDir / "Stamps.csv")
    Tiles = []
    for S in Stamps.head(Cows).itertuples():
        C = Crops[Crops.TrackId == S.StampId].head(PerCow)
        Label = f"{S.Turn} st{S.Stall} {'' if pd.isna(S.CowId) else int(S.CowId)}"
        Row = [_Tile(OutputDir / P, TileSize, Label) for P in C.CropPath]
        Tiles += Row + [np.full((TileSize, TileSize, 3), 30, np.uint8)] * (PerCow - len(Row))
    Out = OutputDir / "StampSheet.jpg"
    cv2.imwrite(str(Out), _Sheet(Tiles, PerCow * 4, TileSize))
    return Out
