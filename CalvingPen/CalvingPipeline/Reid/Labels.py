"""Labelling sheets: what a person needs to name the tracks of a video by eye.

``TrackSheets`` draws one contact sheet per track (its canonical crops over
time) and an overview (one crop per track). ``SuggestGroups`` proposes which
tracks are the same cow (fingerprints alike, never seen at the same time), so
the labeller checks groups instead of single tracks. ``WriteLabelFile``
writes ``Labels.csv``:

    TrackId, FirstTime, LastTime, Crops, GatedCrops, NightShare, Suggested, Cow, Note

Fill ``Cow`` with a letter per animal (A, B, C, ...; the same letter for the
same cow in both parts), ``new`` (test video only: a cow not in the gallery),
``X`` (not a cow, or two cows in one mask) or leave it empty / ``?`` (not sure:
left out). Rerunning keeps what was already filled in.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ..VideoTools import FormatTime
from .Identity import CropScores


def _Tile(Path_, Size, Text):
    Image_ = cv2.imread(str(Path_))
    Image_ = cv2.resize(Image_, (Size, Size)) if Image_ is not None else np.full((Size, Size, 3), 60, np.uint8)
    cv2.rectangle(Image_, (0, 0), (Size, 16), (0, 0, 0), -1)
    cv2.putText(Image_, Text, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
    return Image_


def _Sheet(Tiles, Columns, Size):
    Rows = int(np.ceil(len(Tiles) / Columns))
    Sheet = np.full((Rows * Size, Columns * Size, 3), 30, np.uint8)
    for I, T in enumerate(Tiles):
        Y, X = divmod(I, Columns)
        Sheet[Y * Size:(Y + 1) * Size, X * Size:(X + 1) * Size] = T
    return Sheet


def TrackSheets(VideoDir, Crops=None, MaxTiles=40, Columns=8, TileSize=150, MinCrops=2) -> Path:
    """``Sheets/Track<id>.jpg`` per track with >= ``MinCrops`` crops (at most
    ``MaxTiles``, spread over the track) and ``Sheets/Overview<n>.jpg``."""
    VideoDir = Path(VideoDir)
    Crops = Crops if Crops is not None else pd.read_csv(VideoDir / "Crops.csv")
    Out = VideoDir / "Sheets"
    Out.mkdir(exist_ok=True)
    for Old in Out.glob("*.jpg"):
        Old.unlink()
    Overview = []
    for TrackId, Track in Crops.sort_values("TimeSeconds").groupby("TrackId"):
        if len(Track) < MinCrops:
            continue
        Pick = Track.iloc[np.unique(np.linspace(0, len(Track) - 1, min(MaxTiles, len(Track))).round().astype(int))]
        Tiles = [_Tile(VideoDir / P, TileSize, FormatTime(T)) for P, T in zip(Pick.CropPath, Pick.TimeSeconds)]
        cv2.imwrite(str(Out / f"Track{TrackId:05d}.jpg"), _Sheet(Tiles, Columns, TileSize))
        Middle = Track.iloc[len(Track) // 2]
        Overview.append(_Tile(VideoDir / Middle.CropPath, TileSize,
                              f"T{TrackId} {FormatTime(Track.TimeSeconds.min())[:5]}-{FormatTime(Track.TimeSeconds.max())[:5]}"))
    for N in range(0, len(Overview), 60):
        cv2.imwrite(str(Out / f"Overview{N // 60 + 1}.jpg"), _Sheet(Overview[N:N + 60], 10, TileSize))
    print(f"{len(Overview)} track sheets in {Out}")
    return Out


def SuggestGroups(Crops, Bands, Threshold=0.70, MinCrops=2, TopK=5) -> pd.Series:
    """Suggested letter per track: tracks are merged (average similarity of
    their crops, the matcher's score) while the best pair scores >= ``Threshold``
    and no two tracks of a group are ever seen at the same time. A suggestion
    only: the labeller decides."""
    Gated = Crops[Crops.get("Gated", True)] if "Gated" in Crops else Crops
    Counts = Gated.TrackId.value_counts()
    Ids = sorted(Counts[Counts >= MinCrops].index)
    if len(Ids) < 2:
        return pd.Series({T: "S1" for T in Ids}, dtype=object)
    GatedRows = set(Gated.index)
    Rows = {T: [R for R in np.flatnonzero(Crops.TrackId.to_numpy() == T) if R in GatedRows] for T in Ids}
    Spans = Crops.groupby("TrackId").TimeSeconds.agg(["min", "max"]).loc[Ids]
    # one score per pair of tracks: each track's crops against the other's exemplars
    AllRows = np.concatenate([Rows[T] for T in Ids])
    Owner = np.concatenate([[T] * len(Rows[T]) for T in Ids])
    Scores = CropScores(Bands[AllRows], Bands[AllRows], Owner, Ids, TopK=min(TopK, MinCrops))
    Pair = np.array([np.nanmean(Scores[Owner == T], 0) for T in Ids])
    Pair = np.nan_to_num((Pair + Pair.T) / 2, nan=-1.0)
    # average-linkage merging; tracks seen at the same time never end up in one group
    Lo, Hi = Spans["min"].to_numpy(), Spans["max"].to_numpy()
    Clash = (Lo[:, None] <= Hi[None, :]) & (Lo[None, :] <= Hi[:, None])
    Size, Members, Alive = np.ones(len(Ids)), [[T] for T in Ids], np.ones(len(Ids), bool)
    np.fill_diagonal(Pair, -np.inf)
    while True:
        Masked = np.where(Clash | ~Alive[:, None] | ~Alive[None, :], -np.inf, Pair)
        I, J = np.unravel_index(int(np.argmax(Masked)), Masked.shape)
        if Masked[I, J] < Threshold:
            break
        Pair[I] = Pair[:, I] = (Size[I] * Pair[I] + Size[J] * Pair[J]) / (Size[I] + Size[J])
        Pair[I, I] = -np.inf
        Clash[I] = Clash[:, I] = Clash[I] | Clash[J]
        Size[I] += Size[J]
        Members[I] += Members[J]
        Alive[J] = False
    Groups = [Members[I] for I in np.flatnonzero(Alive)]
    Groups.sort(key=lambda G: -sum(len(Rows[T]) for T in G))
    return pd.Series({T: f"S{N + 1}" for N, G in enumerate(Groups) for T in G}, dtype=object)


def WriteLabelFile(VideoDir, Crops, Suggested=None) -> Path:
    VideoDir = Path(VideoDir)
    Path_ = VideoDir / "Labels.csv"
    Table = (Crops.groupby("TrackId")
             .agg(FirstSeconds=("TimeSeconds", "min"), LastSeconds=("TimeSeconds", "max"), Crops=("CropPath", "count"),
                  GatedCrops=("Gated", "sum") if "Gated" in Crops else ("CropPath", "count"),
                  NightShare=("Night", "mean"))
             .reset_index())
    Table["FirstTime"] = Table.FirstSeconds.map(FormatTime)
    Table["LastTime"] = Table.LastSeconds.map(FormatTime)
    Table["NightShare"] = Table.NightShare.round(2)
    Table["Suggested"] = Table.TrackId.map(Suggested) if Suggested is not None else ""
    Table["Cow"], Table["Note"] = "", ""
    if Path_.exists():                                     # keep what was already filled in
        Old = pd.read_csv(Path_, dtype={"Cow": str, "Note": str}, keep_default_na=False)
        Keep = Old.set_index("TrackId")
        for Column in ("Cow", "Note"):
            Table[Column] = Table.TrackId.map(Keep[Column]).fillna("")
    Columns = ["TrackId", "FirstTime", "LastTime", "Crops", "GatedCrops", "NightShare", "Suggested", "Cow", "Note"]
    Table.sort_values(["Suggested", "FirstSeconds"])[Columns].to_csv(Path_, index=False)
    Filled = (Table.Cow.astype(str).str.strip() != "").sum()
    print(f"{Path_}: {len(Table)} tracks, {Filled} labelled")
    return Path_


def ReadLabels(VideoDir) -> dict:
    """TrackId -> label (A, B, ..., 'new', 'X'); empty and '?' are left out."""
    Table = pd.read_csv(Path(VideoDir) / "Labels.csv", dtype={"Cow": str}, keep_default_na=False)
    Labels = {}
    for T, C in zip(Table.TrackId, Table.Cow):
        C = str(C).strip()
        if C and C != "?":
            Labels[int(T)] = "new" if C.lower() == "new" else "X" if C.upper() == "X" else C.upper()
    return Labels
