"""Validation protocol v1: gallery = part 1 of a day (labelled), test = part 2.

Rules (leakage guards): no test crop is ever in the anchor pool; the
settings (``Tau``, ``AnchorWeight``, ``NewScore``) are chosen on the gallery
only (``TuneOnGallery``: first half names the second half), saved in
``Tuned.json`` and frozen before the test video is named; the test video's own
labels are only used for scoring.

``ScoreNames`` counts cow-minutes (each crop stands for ``CropSeconds`` of one
cow): accuracy of named time = correct / (correct + wrong); "?" time; wrong
names split into silent and flagged; new-cow rejection (share of the time of
cows not in the gallery called "?" / "new"); and the same per night/day and
for the first ``SeamMinutes`` vs the rest (a cow standing still across the
seam is matched trivially).
"""

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .Bands import LoadBands
from .Identity import GateCrops, NameTracks
from .Labels import ReadLabels
from .PenArea import InPen

Defaults = {"WindowMinutes": 3, "Tau": 0.015, "AnchorWeight": 0.6, "LagMinutes": 30, "MinCrops": 2, "NewScore": None}


def LoadVideo(VideoDir, PenArea=None, **Gates):
    """(gated crops incl. NonCow tracks, their bands, Tracks.csv) of one video.
    ``PenArea``: polygon of the pen floor (``PenArea.PenAreaFor``); crops of
    cows in the next pens are dropped first, so they get no sheet, no label and
    no name."""
    Crops, Bands = LoadBands(VideoDir)
    if PenArea is not None:
        Keep = InPen(Crops, PenArea).to_numpy()
        Crops, Bands = Crops[Keep].reset_index(drop=True), Bands[Keep]
    Crops = GateCrops(Crops, **Gates)
    Tracks = pd.read_csv(Path(VideoDir) / "Tracks.csv", usecols=["TrackId", "TimeSeconds"])
    return Crops, Bands, Tracks


def Anchors(Crops, Bands, Labels):
    """Gallery crops that can be anchors: gated, of a labelled cow (not X / new)."""
    Cow = Crops.TrackId.map(Labels)
    Keep = (Crops.Gated & ~Crops.NonCow & Cow.notna() & ~Cow.isin(["X", "new"])).to_numpy()
    return Crops[Keep].assign(Cow=Cow[Keep]).reset_index(drop=True), Bands[Keep]


def NameVideo(Crops, Bands, Tracks, AnchorCrops, AnchorBands, OutputDir=None, **Settings):
    """Names every track of a video. NonCow tracks are "not a cow". Writes
    TrackNames.csv, CropNames.csv and Windows.csv when ``OutputDir`` is given."""
    Settings = {**Defaults, **Settings}
    Query = Crops.Gated & ~Crops.NonCow
    Names, CropNames, Windows = NameTracks(Crops[Query], Bands[Query.to_numpy()], AnchorCrops, AnchorBands,
                                           Tracks=Tracks, **Settings)
    Others = Crops[~Crops.TrackId.isin(Names.TrackId)]
    Extra = (Others.groupby("TrackId").agg(Crops=("CropPath", "count"), FirstSeconds=("TimeSeconds", "min"),
                                           LastSeconds=("TimeSeconds", "max"), NonCow=("NonCow", "max")).reset_index())
    Extra["Name"] = np.where(Extra.NonCow, "not a cow", "?")
    Extra["Flags"] = np.where(Extra.NonCow, "coloured (staff?)", "no usable crop")
    Names = pd.concat([Names, Extra.drop(columns="NonCow")], ignore_index=True).sort_values("FirstSeconds")
    AllCrops = Crops.merge(Names[["TrackId", "Name", "Flags"]], on="TrackId", how="left")
    if OutputDir:
        OutputDir = Path(OutputDir)
        OutputDir.mkdir(parents=True, exist_ok=True)
        Names.to_csv(OutputDir / "TrackNames.csv", index=False)
        AllCrops.drop(columns=[C for C in AllCrops if C.startswith(("X1", "Y1", "X2", "Y2"))]).to_csv(
            OutputDir / "CropNames.csv", index=False)
        Windows.to_csv(OutputDir / "Windows.csv", index=False)
        (OutputDir / "Settings.json").write_text(json.dumps(Settings, indent=1))
    return Names, AllCrops, Windows


def ScoreNames(CropNames, Truth, GalleryCows, CropSeconds=15, SeamMinutes=10, VideoStartSeconds=None) -> dict:
    """Cow-minute scores of the names against the labels (``Truth``: TrackId -> label)."""
    Rows = CropNames[CropNames.Gated & ~CropNames.NonCow].copy() if "Gated" in CropNames else CropNames.copy()
    Rows["Truth"] = Rows.TrackId.map(Truth)
    Rows = Rows[Rows.Truth.notna() & (Rows.Truth != "X")]
    Rows["Flagged"] = Rows.Flags.fillna("").astype(str) != ""
    Start = VideoStartSeconds if VideoStartSeconds is not None else (Rows.TimeSeconds.min() if len(Rows) else 0)
    Rows["Part"] = np.where(Rows.TimeSeconds < Start + SeamMinutes * 60, f"first {SeamMinutes} min", "rest")
    Minutes = CropSeconds / 60

    def Summary(R):
        Known = R[R.Truth.isin(GalleryCows)]
        Named = Known[Known.Name.isin(GalleryCows)]
        Right = int((Named.Name == Named.Truth).sum())
        Wrong = Named[Named.Name != Named.Truth]
        New = R[R.Truth == "new"]
        return {"CowMinutes": round(len(R) * Minutes, 1),
                "Accuracy": round(Right / len(Named), 4) if len(Named) else None,
                "NamedMinutes": round(len(Named) * Minutes, 1),
                "UnsureMinutes": round(int(Known.Name.isin(["?", "new"]).sum()) * Minutes, 1),
                "SilentErrorMinutes": round(int((~Wrong.Flagged).sum()) * Minutes, 1),
                "FlaggedErrorMinutes": round(int(Wrong.Flagged.sum()) * Minutes, 1),
                "NewCowMinutes": round(len(New) * Minutes, 1),
                "NewCowRejection": round(float(New.Name.isin(["?", "new"]).mean()), 4) if len(New) else None}
    Out = {"All": Summary(Rows)}
    for Key, Group in itertools.chain(Rows.groupby("Part"), Rows.groupby(np.where(Rows.Night, "night", "day"))):
        Out[str(Key)] = Summary(Group)
    return Out


def TuneOnGallery(Crops, Bands, Tracks, Labels, Grid=None, CropSeconds=15) -> tuple:
    """Choose the settings on the gallery only: its first half (by time) is the
    anchor pool, its second half is named and scored. Cows only seen in the
    second half count as "new". Returns (best settings, table of all tried)."""
    # split by track (its middle time), so no track has crops on both sides
    Middle = Crops.groupby("TrackId").TimeSeconds.median()
    Split = Crops.TimeSeconds.median()
    First = Crops.TrackId.map(Middle <= Split).astype(bool)
    Second = ~First
    FirstLabels = {T: L for T, L in Labels.items() if T in set(Crops.TrackId[First])}
    AnchorCrops, AnchorBands = Anchors(Crops[First].reset_index(drop=True), Bands[First.to_numpy()], FirstLabels)
    Cows = sorted(AnchorCrops.Cow.unique())
    if not Cows:
        raise ValueError("No labelled cow in the first half of the gallery: label Labels.csv first")
    Truth = {T: (L if L in Cows or L in ("X", "new") else "new") for T, L in Labels.items()}
    Test = Crops[Second].reset_index(drop=True)
    TestBands = Bands[Second.to_numpy()]
    # NewScore candidates: low quantiles of the best anchor score of crops of known cows
    from .Identity import CropScores
    Known = Test.Gated & Test.TrackId.map(Truth).isin(Cows)
    Best = np.nanmax(CropScores(TestBands[Known.to_numpy()], AnchorBands, AnchorCrops.Cow, Cows), 1) if Known.any() else []
    NewScores = [None] + ([round(float(np.nanquantile(Best, Q)), 4) for Q in (0.01, 0.05, 0.15)] if len(Best) else [])
    Grid = Grid or {"Tau": [0.005, 0.015, 0.03], "AnchorWeight": [0.6, 1.0], "NewScore": NewScores}
    Results = []
    for Values in itertools.product(*Grid.values()):
        Settings = dict(zip(Grid.keys(), Values))
        _, CropNames, _ = NameVideo(Test, TestBands, Tracks, AnchorCrops, AnchorBands, **Settings)
        Score = ScoreNames(CropNames, Truth, Cows, CropSeconds)["All"]
        Results.append({**Settings, **Score})
    Table = pd.DataFrame(Results)
    # best: correct named time, minus wrong time (silent errors count double), then new-cow rejection
    Table["Utility"] = (Table.Accuracy.fillna(0) * Table.NamedMinutes - Table.FlaggedErrorMinutes
                        - 2 * Table.SilentErrorMinutes).round(1)
    Table = Table.sort_values(["Utility", "NewCowRejection"], ascending=[False, False],
                              na_position="last").reset_index(drop=True)
    Pick = 0
    if "NewScore" in Grid and not (Table.NewCowMinutes > 0).any():
        # no unknown cow to tune the "new" threshold on: take the strictest one that costs <= 1 % of the best result
        Near = Table[(Table.Utility >= 0.99 * Table.Utility.iloc[0]) & Table.NewScore.notna()]
        if len(Near):
            Pick = Near.NewScore.astype(float).idxmax()
    BestSettings = {**Defaults, **{K: (None if pd.isna(Table.loc[Pick, K]) else Table.loc[Pick, K]) for K in Grid}}
    BestSettings = {K: (float(V) if isinstance(V, (np.floating, float)) else V) for K, V in BestSettings.items()}
    return BestSettings, Table
