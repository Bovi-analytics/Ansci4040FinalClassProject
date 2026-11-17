"""Step 06: put the fingerprints of all videos into one labelled pool.

Every video folder under ``OutputRoot`` that has step 05 output contributes
its fingerprints. Each track gets an identity (which cow it is) from the
video's ``Identities.csv``:

    TrackId, IdentityId, Use, Note
    1, Pen2Cow1, True,
    2, Pen2Cow2, True, second cow
    ...

Cows are labelled per pen, numbered in date order: ``Pen2Cow1``, ``Pen2Cow2``,
``Pen3Cow1``, ... (``CowRegister.csv`` next to the video folders remembers which
pen, day and cow-in-video each label is). On first run every track gets the
cow of the video's name (``CalvingPen2Nov11Cow2`` -> cow 2 of pen 2 on Nov 11),
which becomes e.g. ``Pen2Cow4``. For another cow in the same video write
``Cow3`` etc. (cow 3 of that pen and day), or write a full label (``Pen2Cow1``,
or a farm ID) to say a track is a cow already known. ``Use`` = False leaves a track out (e.g. the calf).
Unlike the step outputs, this file is **never overwritten**: new tracks are
appended, your edits are kept.

Outputs in ``PoolDir``, overwritten on every run:

- ``PoolEmbeddings.npy`` + ``PoolIndex.csv``: one row per crop (VideoStem,
  TrackId, IdentityId, CropPath, ...), near-duplicate crops left out
- ``PoolSummary.csv``: per identity, how many videos, tracks and crops
- ``SelfCheck.csv`` / ``SelfCheck.png``: for each crop, its closest crop in
  *another* track (as if that track were new; with ``ExcludeSeconds``, also
  its own track more than that many seconds away) and whether it is the same
  cow. With one cow this shows how high her true matches score; with more
  cows it is the first accuracy test and shows where a threshold can go
- ``Model.txt``: the model all fingerprints came from (they must match)
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

IdentityColumns = ["TrackId", "IdentityId", "Use", "Note"]


def ParseVideo(VideoStem: str):
    """(Pen, "Nov11", CowInVideo) of a video folder name, old or new style, or None.
    CowInVideo is the ``Cow<n>`` of the name (``CalvingPen3Nov11Cow2`` -> 2), else 1."""
    from .Rename import CalvingName, Months
    Name = CalvingName(VideoStem)[0]                                   # any style -> CalvingPen3Nov11Cow2[ROI][Win1]
    Hit = re.match(r"CalvingPen(\d+)(" + "|".join(Months) + r")(\d{1,2})Cow(\d+)", Name)
    if Hit:
        return int(Hit.group(1)), f"{Hit.group(2)}{int(Hit.group(3))}", int(Hit.group(4))
    return None


def _DayOrder(Day: str) -> tuple:
    from .Rename import Months
    return (Months.index(Day[:3]), int(Day[3:])) if Day and Day[:3] in Months else (99, 99)


def LoadRegister(OutputRoot) -> pd.DataFrame:
    """``CowRegister.csv``: which cow label belongs to which pen, day and cow-in-video number."""
    Path_ = Path(OutputRoot) / "CowRegister.csv"
    if Path_.exists():
        return pd.read_csv(Path_, dtype={"Day": str, "IdentityId": str, "FirstVideo": str})
    return pd.DataFrame(columns=["Pen", "Day", "LocalCow", "IdentityId", "FirstVideo"])


def CowLabel(Register: pd.DataFrame, Pen, Day, LocalCow, VideoStem):
    """``Pen<pen>Cow<n>`` for cow ``LocalCow`` of that pen and day: the label it already
    has in the register, else the next free number in that pen (Pen2Cow1, Pen2Cow2, ...)."""
    Hit = Register[(Register.Pen.astype(int) == int(Pen)) & (Register.Day == Day) & (Register.LocalCow.astype(int) == int(LocalCow))]
    if len(Hit):
        return Hit.IdentityId.iloc[0], Register
    Used = Register.IdentityId[Register.Pen.astype(int) == int(Pen)].str.extract(r"Cow(\d+)$")[0].dropna().astype(int)
    Label = f"Pen{int(Pen)}Cow{(Used.max() if len(Used) else 0) + 1}"
    Row = pd.DataFrame([{"Pen": int(Pen), "Day": Day, "LocalCow": int(LocalCow), "IdentityId": Label, "FirstVideo": VideoStem}])
    return Label, pd.concat([Register, Row], ignore_index=True)


def ResolveLabel(Text: str, VideoStem: str, Register):
    """Turn what is written in Identities.csv into the final cow label:
    ``Cow2`` (2nd cow of this video) or an old ``Pen2_20260926_Cow01`` -> ``Pen2Cow3``-style;
    ``Pen2Cow3`` or anything else (e.g. a farm ID) is kept as written."""
    from .Rename import MonthDay
    Text = str(Text).strip()
    Old = re.fullmatch(r"Pen(\d+)_(\d{4})(\d{2})(\d{2})_Cow(\d+)", Text)
    if Old:
        return CowLabel(Register, Old.group(1), MonthDay(*Old.groups()[1:4]), Old.group(5), VideoStem)
    Short = re.fullmatch(r"Cow\s*(\d+)", Text, flags=re.I)
    Video = ParseVideo(VideoStem)
    if Short and Video:
        return CowLabel(Register, Video[0], Video[1], Short.group(1), VideoStem)
    if Text == "" and Video:                                         # new track: the cow of the file name
        return CowLabel(Register, Video[0], Video[1], Video[2], VideoStem)
    if Text == "":
        return re.sub(r"\W+", "", VideoStem.title()) + "Cow1", Register
    if Short:
        return re.sub(r"\W+", "", VideoStem.title()) + f"Cow{Short.group(1)}", Register
    return Text, Register


def LoadIdentities(VideoDir: Path, TrackIds, Register):
    """Read ``Identities.csv``, add tracks it does not list yet (as ``Cow1``: one cow
    per video until relabelled) and turn every label into its final form
    (Pen<pen>Cow<n>). Returns (table, register)."""
    IdentityPath = Path(VideoDir) / "Identities.csv"
    if IdentityPath.exists():
        Table = pd.read_csv(IdentityPath, dtype={"IdentityId": str, "Note": str}, keep_default_na=False)
        Table["Use"] = Table["Use"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
    else:
        Table = pd.DataFrame(columns=IdentityColumns)
    New = sorted(set(int(T) for T in TrackIds) - set(Table.TrackId.astype(int)))
    if New:
        Table = pd.concat([Table, pd.DataFrame({"TrackId": New, "IdentityId": "", "Use": True, "Note": ""})])
    Labels = []
    for Text in Table.IdentityId:
        Label, Register = ResolveLabel(Text, Path(VideoDir).name, Register)
        Labels.append(Label)
    Changed = New or Labels != list(Table.IdentityId)
    Table["IdentityId"] = Labels
    Table = Table.astype({"TrackId": int}).sort_values("TrackId")
    if Changed:
        Table[IdentityColumns].to_csv(IdentityPath, index=False)
        print(f"  {IdentityPath.name}: {', '.join(f'T{T}={L}' for T, L in zip(Table.TrackId, Table.IdentityId))}")
    return Table, Register


def FindVideos(OutputRoot, EmbeddingFolder) -> list:
    return sorted(P.parent.parent for P in Path(OutputRoot).glob(f"*/{EmbeddingFolder}/Embeddings.npy"))


def SelfCheck(Vectors, Index: pd.DataFrame, ExcludeSeconds=None) -> pd.DataFrame:
    """Closest crop from another track for every crop (leave-one-track-out).
    With ``ExcludeSeconds``, only crops of the same track within that many
    seconds are hidden, so a one-hour SAM track can be checked against its
    own crops from much earlier or later."""
    Similarity = Vectors @ Vectors.T
    Track = (Index.VideoStem + "/" + Index.TrackId.astype(str)).to_numpy(dtype=object)
    Hide = Track[:, None] == Track[None, :]
    if ExcludeSeconds:
        Times = Index.TimeSeconds.to_numpy(dtype=float)
        Hide &= np.abs(Times[:, None] - Times[None, :]) < ExcludeSeconds
    Similarity[Hide] = -np.inf   # never match your own track (or its nearby crops)
    Identity = Index.IdentityId.to_numpy(dtype=object)
    Rows = []
    for I in range(len(Index)):
        Best = int(np.argmax(Similarity[I]))
        if not np.isfinite(Similarity[I, Best]):
            continue                                           # only one track in the pool
        Same, Other = Identity == Identity[I], Identity != Identity[I]
        SameRow, OtherRow = Similarity[I].copy(), Similarity[I].copy()
        SameRow[~Same], OtherRow[~Other] = -np.inf, -np.inf
        Rows.append({
            "VideoStem": Index.VideoStem.iat[I], "TrackId": Index.TrackId.iat[I],
            "IdentityId": Identity[I], "TimeSeconds": Index.TimeSeconds.iat[I], "CropPath": Index.CropPath.iat[I],
            "BestMatchIdentity": Identity[Best], "BestMatchTrack": Index.TrackId.iat[Best],
            "BestMatchSeconds": Index.TimeSeconds.iat[Best],
            "BestSimilarity": round(float(Similarity[I, Best]), 3),
            "BestSameCow": round(float(SameRow.max()), 3) if np.isfinite(SameRow.max()) else np.nan,
            "BestOtherCow": round(float(OtherRow.max()), 3) if np.isfinite(OtherRow.max()) else np.nan,
            "Correct": Identity[Best] == Identity[I],
        })
    return pd.DataFrame(Rows)


def _SaveSelfCheckPlot(Check: pd.DataFrame, OutPath, Title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Fig, Ax = plt.subplots(figsize=(7, 3.8))
    Bins = np.linspace(0, 1, 51)
    Ax.hist(Check.BestSameCow.dropna(), bins=Bins, alpha=0.7, label="closest crop of the SAME cow (other track)")
    if Check.BestOtherCow.notna().any():
        Ax.hist(Check.BestOtherCow.dropna(), bins=Bins, alpha=0.7, label="closest crop of a DIFFERENT cow")
    Ax.set_xlabel("cosine similarity")
    Ax.set_ylabel("crops")
    Ax.set_title(Title, fontsize=10)
    Ax.legend(fontsize=8)
    Fig.tight_layout()
    Fig.savefig(OutPath, dpi=120)
    plt.close(Fig)


def BuildPool(OutputRoot, PoolDir, EmbeddingFolder="Step05bEmbedFull", VideoStems=None, SkipNearDuplicates=True,
              ExcludeSeconds=None) -> dict:
    OutputRoot, PoolDir = Path(OutputRoot), Path(PoolDir)
    PoolDir.mkdir(parents=True, exist_ok=True)
    Videos = FindVideos(OutputRoot, EmbeddingFolder)
    if VideoStems:
        Videos = [V for V in Videos if V.name in set(VideoStems)]
    if not Videos:
        raise FileNotFoundError(f"No */{EmbeddingFolder}/Embeddings.npy under {OutputRoot}")

    # Oldest recordings first, so cows are numbered in date order (Pen2Cow1 = first cow in pen 2).
    Videos = sorted(Videos, key=lambda V: _DayOrder((ParseVideo(V.name) or (0, "", 0))[1]) + (V.name,))
    Register = LoadRegister(OutputRoot)
    AllVectors, AllIndex, Models = [], [], {}
    for VideoDir in Videos:
        Folder = VideoDir / EmbeddingFolder
        Vectors = np.load(Folder / "Embeddings.npy")
        Index = pd.read_csv(Folder / "EmbeddingIndex.csv")
        Models[VideoDir.name] = (Folder / "Model.txt").read_text().strip()
        Identities, Register = LoadIdentities(VideoDir, Index.TrackId.unique(), Register)
        Index = Index.merge(Identities[["TrackId", "IdentityId", "Use"]], on="TrackId", how="left")
        Keep = Index.Use.astype(bool).to_numpy().copy()
        if SkipNearDuplicates and "NearDuplicate" in Index:
            Keep &= ~Index.NearDuplicate.astype(bool).values
        print(f"{VideoDir.name}: {Keep.sum()} of {len(Index)} crops, "
              f"{Index[Keep].TrackId.nunique()} tracks, {Models[VideoDir.name]}")
        AllVectors.append(Vectors[Keep])
        AllIndex.append(Index[Keep])

    Register.to_csv(OutputRoot / "CowRegister.csv", index=False)
    if len(set(Models.values())) > 1:
        raise ValueError(f"Fingerprints come from different models, cannot pool them: {Models}. "
                         "Rerun step 05 so every video uses the same model.")

    Vectors = np.concatenate(AllVectors)
    Index = pd.concat(AllIndex, ignore_index=True)[
        ["VideoStem", "TrackId", "IdentityId", "Frame", "TimeSeconds", "CropPath"]]
    np.save(PoolDir / "PoolEmbeddings.npy", Vectors.astype(np.float32))
    Index.to_csv(PoolDir / "PoolIndex.csv", index=False)
    (PoolDir / "Model.txt").write_text(next(iter(Models.values())) + "\n")

    Tracks = Index[["IdentityId", "VideoStem", "TrackId"]].drop_duplicates().groupby("IdentityId").size()
    Summary = (Index.groupby("IdentityId").agg(Videos=("VideoStem", "nunique"), Crops=("CropPath", "count"))
               .assign(Tracks=Tracks)[["Videos", "Tracks", "Crops"]].reset_index())
    Summary.to_csv(PoolDir / "PoolSummary.csv", index=False)

    (PoolDir / "SelfCheck.png").unlink(missing_ok=True)
    Check = SelfCheck(Vectors, Index, ExcludeSeconds)
    Check.to_csv(PoolDir / "SelfCheck.csv", index=False)
    if len(Check):
        _SaveSelfCheckPlot(Check, PoolDir / "SelfCheck.png",
                           f"{len(Summary)} cow(s), {len(Index)} crops: best match in another track")

    Stats = {"Videos": len(Videos), "Identities": len(Summary), "Crops": len(Index),
             "Model": next(iter(Models.values()))}
    if len(Check):
        Stats["SameCowBestMedian"] = round(float(Check.BestSameCow.median()), 3)
        Stats["SameCowBestMin"] = round(float(Check.BestSameCow.min()), 3)
        if Check.BestOtherCow.notna().any():
            Stats["OtherCowBestMedian"] = round(float(Check.BestOtherCow.median()), 3)
            Stats["Top1Accuracy"] = round(float(Check.Correct.mean()), 3)
    print(Stats)
    return Stats
