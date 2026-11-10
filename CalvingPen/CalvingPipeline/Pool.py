"""Step 06: put the fingerprints of all videos into one labelled pool.

Every video folder under ``OutputRoot`` that has step 05 output contributes
its fingerprints. Each track gets an identity (which cow it is) from the
video's ``Identities.csv``:

    TrackId, IdentityId, Use, Note
    2, Pen2_20260926_Cow01, True,
    234, Pen2_20260926_Cow01, True, rear view
    ...

The file is created on first run with every track set to the video's one
default cow (a calving pen usually holds one cow), e.g.
``Pen2_20260926_Cow01``. Edit it in Drive / Google Sheets to give a track
another cow, or set ``Use`` to False to leave a track out (e.g. the calf).
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


def DefaultIdentity(VideoStem: str) -> str:
    """``2026-09-26 00-00-00 Calving Pen 2`` -> ``Pen2_20260926_Cow01``."""
    Match = re.search(r"(\d{4})-(\d{2})-(\d{2}).*?Pen\s*(\d+)", VideoStem)
    if Match:
        Year, Month, Day, Pen = Match.groups()
        return f"Pen{Pen}_{Year}{Month}{Day}_Cow01"
    return re.sub(r"\W+", "", VideoStem.title()) + "_Cow01"


def LoadIdentities(VideoDir: Path, TrackIds) -> pd.DataFrame:
    """Read ``Identities.csv``, adding any track it does not list yet."""
    IdentityPath = Path(VideoDir) / "Identities.csv"
    if IdentityPath.exists():
        Table = pd.read_csv(IdentityPath, dtype={"IdentityId": str, "Note": str}, keep_default_na=False)
        Table["Use"] = Table["Use"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
    else:
        Table = pd.DataFrame(columns=IdentityColumns)
    New = sorted(set(int(T) for T in TrackIds) - set(Table.TrackId.astype(int)))
    if New:
        Default = DefaultIdentity(Path(VideoDir).name)
        Table = pd.concat([Table, pd.DataFrame({"TrackId": New, "IdentityId": Default, "Use": True, "Note": ""})])
        Table = Table.astype({"TrackId": int}).sort_values("TrackId")
        Table[IdentityColumns].to_csv(IdentityPath, index=False)
        print(f"  {IdentityPath.name}: added {len(New)} tracks as {Default}")
    return Table.astype({"TrackId": int})


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

    AllVectors, AllIndex, Models = [], [], {}
    for VideoDir in Videos:
        Folder = VideoDir / EmbeddingFolder
        Vectors = np.load(Folder / "Embeddings.npy")
        Index = pd.read_csv(Folder / "EmbeddingIndex.csv")
        Models[VideoDir.name] = (Folder / "Model.txt").read_text().strip()
        Identities = LoadIdentities(VideoDir, Index.TrackId.unique())
        Index = Index.merge(Identities[["TrackId", "IdentityId", "Use"]], on="TrackId", how="left")
        Keep = Index.Use.astype(bool).to_numpy().copy()
        if SkipNearDuplicates and "NearDuplicate" in Index:
            Keep &= ~Index.NearDuplicate.astype(bool).values
        print(f"{VideoDir.name}: {Keep.sum()} of {len(Index)} crops, "
              f"{Index[Keep].TrackId.nunique()} tracks, {Models[VideoDir.name]}")
        AllVectors.append(Vectors[Keep])
        AllIndex.append(Index[Keep])

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
