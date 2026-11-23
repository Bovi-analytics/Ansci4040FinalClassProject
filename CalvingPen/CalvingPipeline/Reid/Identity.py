"""Name every track segment (stages 6-9 of the re-ID design), offline, after
tracking. Simplified port of the matcher and identity layer in
Tol_30_periodic_identity.py / Tol_50 of Sonam525/Re-id-for-tracking (MIT licence).

Gates (``GateCrops``), per crop: mask area >= ``GateArea``, no other cow on
more than 10 % of it (``NeighbourIos``), not over-extended (> 1.5 x the
track's median area, i.e. two cows in one mask), not clearly coloured. A track
whose daytime crops are mostly coloured is marked NonCow (staff) and never named.

Matcher (``CropScores``): a query crop's score for a cow = the mean of its 5
best similarities to that cow's individual exemplars (not their average), on
the 3 body thirds, the better of the two orientations.

Identity layer (``NameTracks``), every ``WindowMinutes``:

1. each live track (>= ``MinCrops`` gated crops in the window) gets a row of
   scores, one per gallery cow: ``AnchorWeight`` x anchor pool (the labelled
   gallery crops, frozen) + the rest x temporal pool (crops of this video
   already named confidently, at least ``LagMinutes`` older). While a cow has
   fewer than 5 temporal exemplars, the anchors alone are used.
2. the rows are added up per track (all its windows so far), and one joint
   assignment (Hungarian) over the live tracks names them, so two tracks seen
   at the same time never get the same cow; a cow held by a track that was
   ever seen at the same time as this one is excluded too.
3. a name changes only when the new assignment is better by ``Tau``; else the
   tracks are marked contested. The name holds for the whole track (backward
   and forward).
4. the track's final name is "?" when its best cow beats the second by less
   than ``Tau``, and "new" when no cow reaches ``NewScore`` (a cow not in the
   gallery).
"""

from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

P3, P3R = [1, 2, 3], [5, 6, 7]
Forbidden = -1e6


def GateCrops(Crops, GateArea=20000, MaxNeighbourIos=0.10, AreaOver=1.5, MaxColoured=0.065, NonCowShare=0.5):
    """Adds ``Gated``, ``GateReason`` per crop and ``NonCow`` per track."""
    Crops = Crops.copy()
    Reasons = [[] for _ in range(len(Crops))]
    Area = Crops.MaskArea.to_numpy(float)
    Median = Crops.assign(A=np.where(Area >= GateArea, Area, np.nan)).groupby("TrackId").A.transform("median")
    Refs = Crops.assign(A=Area >= GateArea).groupby("TrackId").A.transform("sum")
    Coloured = (Crops.ColouredFraction.fillna(0) > MaxColoured) & ~Crops.Night.astype(bool)
    for I, (A, Nb, Md, Rf, Co) in enumerate(zip(Area, Crops.NeighbourIos, Median, Refs, Coloured)):
        if A < GateArea:
            Reasons[I].append("small")
        if Nb >= MaxNeighbourIos:
            Reasons[I].append("touching another cow")
        if Rf >= 10 and A > AreaOver * Md:
            Reasons[I].append("over-extended")
        if Co:
            Reasons[I].append("coloured")
    Crops["GateReason"] = ["; ".join(R) for R in Reasons]
    Crops["Gated"] = Crops.GateReason == ""
    Day = Crops[~Crops.Night.astype(bool)]
    Share = Day.assign(C=Coloured[Day.index]).groupby("TrackId").C.mean()
    Crops["NonCow"] = Crops.TrackId.map(Share >= NonCowShare).fillna(False).astype(bool)
    return Crops


def Parts3(Bands):
    N = len(Bands)
    return (np.ascontiguousarray(Bands[:, P3, :].reshape(N, -1)),
            np.ascontiguousarray(Bands[:, P3R, :].reshape(N, -1)))


def CropScores(QueryBands, PoolBands, PoolCows, Cows, TopK=5, Chunk=1024) -> np.ndarray:
    """[queries, cows]: mean of the ``TopK`` best matches to each cow's
    exemplars (NaN for a cow with fewer than ``TopK``)."""
    Q3, _ = Parts3(QueryBands)
    A3, A3R = Parts3(PoolBands)
    PoolCows = np.asarray(PoolCows)
    Out = np.full((len(Q3), len(Cows)), np.nan, np.float32)
    for C, Cow in enumerate(Cows):
        Sel = np.flatnonzero(PoolCows == Cow)
        if len(Sel) < TopK:
            continue
        X0, X1 = A3[Sel].T, A3R[Sel].T
        for S0 in range(0, len(Q3), Chunk):
            S = np.maximum(Q3[S0:S0 + Chunk] @ X0, Q3[S0:S0 + Chunk] @ X1) / 3.0
            Out[S0:S0 + Chunk, C] = np.sort(np.partition(S, -TopK, axis=1)[:, -TopK:], axis=1).mean(1)
    return Out


def _Spans(Crops, Tracks=None):
    Source = Tracks if Tracks is not None else Crops
    return Source.groupby("TrackId").TimeSeconds.agg(["min", "max"])


def _Margin(Row, NewScore=None):
    """Best cow minus second-best cow (the "new" threshold is a separate test)."""
    Values = sorted([V for V in Row if np.isfinite(V) and V > Forbidden / 1e4], reverse=True)
    return (Values[0] - Values[1]) if len(Values) > 1 else np.inf


def NameTracks(Query, QueryBands, Anchors, AnchorBands, Tracks=None, WindowMinutes=3, Tau=0.015, AnchorWeight=0.6,
               LagMinutes=30, MinCrops=2, NewScore=None, TopK=5, PoolCap=4000):
    """``Query``: the test video's gated crops (``GateCrops``), rows aligned with
    ``QueryBands``; ``Anchors``: gallery crops with a ``Cow`` column, aligned with
    ``AnchorBands``. ``Tracks``: the video's Tracks.csv (better track spans).
    Returns (TrackNames, CropNames, Windows)."""
    Query = Query.reset_index(drop=True)
    Cows = sorted(Anchors.Cow.unique())
    Anchor = CropScores(QueryBands, AnchorBands, Anchors.Cow.to_numpy(), Cows, TopK)
    Spans = _Spans(Query, Tracks)
    Window = (Query.TimeSeconds // (WindowMinutes * 60)).astype(int).to_numpy()
    Sum, Count = defaultdict(lambda: np.zeros(len(Cows))), defaultdict(int)
    Label, Flags, Pool, Records = {}, defaultdict(set), [], []

    def Concurrent(A, B):
        return Spans.loc[A, "min"] <= Spans.loc[B, "max"] and Spans.loc[B, "min"] <= Spans.loc[A, "max"]

    for W in np.unique(Window):
        Rows = np.flatnonzero(Window == W)
        Start = W * WindowMinutes * 60
        ByTrack = pd.Series(Rows).groupby(Query.TrackId.to_numpy()[Rows]).apply(list)
        Live = [T for T, R in ByTrack.items() if len(R) >= MinCrops]
        if not Live:
            continue
        Scores = Anchor[Rows].astype(float)
        Visible = [(I, C) for (Tm, C, I) in Pool if Tm <= Start - LagMinutes * 60]
        Blend = AnchorWeight < 1 and Visible and all(sum(1 for _, C in Visible if C == Cow) >= TopK for Cow in Cows)
        if Blend:
            Idx = np.array([I for I, _ in Visible])
            Temporal = CropScores(QueryBands[Rows], QueryBands[Idx], np.array([C for _, C in Visible]), Cows, TopK)
            Scores = AnchorWeight * Scores + (1 - AnchorWeight) * Temporal
        Local = {R: K for K, R in enumerate(Rows)}
        for T in Live:
            Values = Scores[[Local[R] for R in ByTrack[T]]]
            Sum[T] += np.nan_to_num(np.nanmean(Values, 0), nan=Forbidden / 1e3) * len(Values)
            Count[T] += len(Values)
        # joint decision over the live tracks, on their accumulated rows
        M = np.array([Sum[T] / Count[T] for T in Live])
        for I, T in enumerate(Live):                     # a cow held by a finished track seen together with T
            for Other, Cow in Label.items():
                if Other not in Live and Cow in Cows and Concurrent(T, Other):
                    M[I, Cows.index(Cow)] = Forbidden
        Choices = list(Cows) + (["new"] * len(Live) if NewScore is not None else [])
        Matrix = np.hstack([M, np.full((len(Live), len(Live)), NewScore)]) if NewScore is not None else M
        R_, C_ = linear_sum_assignment(-Matrix)
        Proposed = {Live[R]: (Choices[C] if Matrix[R, C] > Forbidden / 2 else "?") for R, C in zip(R_, C_)}
        Current = {T: Label[T] for T in Live if T in Label}
        Changed = [T for T in Current if Proposed.get(T, "?") != Current[T]]
        if Changed:
            def Joint(Names):
                Total = 0.0
                for I, T in enumerate(Live):
                    N = Names.get(T, "?")
                    Total += (NewScore if N == "new" else M[I, Cows.index(N)] if N in Cows else 0.0)
                return Total
            Held = {**Proposed, **Current}
            Unique = len([N for N in Held.values() if N in Cows]) == len(set(N for N in Held.values() if N in Cows))
            if Unique and Joint(Proposed) - Joint(Held) < Tau:
                Proposed = Held
                for T in Changed:
                    Flags[T].add("contested")
        for T in Live:
            Label[T] = Proposed.get(T, Label.get(T, "?"))
            Margin = _Margin(Sum[T] / Count[T], NewScore)
            Confident = Label[T] in Cows and Margin >= Tau
            if Confident:
                Pool += [(Query.TimeSeconds[R], Label[T], R) for R in ByTrack[T]]
            Records.append({"Window": int(W), "StartSeconds": Start, "TrackId": int(T), "Crops": len(ByTrack[T]),
                            "Name": Label[T], "Margin": round(float(Margin), 4), "Blend": bool(Blend),
                            **{f"Score{C}": round(float(V), 4) for C, V in zip(Cows, Sum[T] / Count[T])}})
        if len(Pool) > PoolCap * len(Cows):             # keep the most recent exemplars per cow
            Pool = sorted(Pool)[-PoolCap * len(Cows):]

    Rows_ = []
    for T, Group in Query.groupby("TrackId"):
        if T in Count:
            Row = Sum[T] / Count[T]
            Name = Label[T]
        else:                                             # never live in any window: anchors only, flagged
            Row = np.nan_to_num(np.nanmean(Anchor[Group.index], 0), nan=Forbidden / 1e3)
            Best = int(np.argmax(Row))
            Name = Cows[Best] if NewScore is None or Row[Best] >= NewScore else "new"
            Flags[T].add("short")
        Margin = _Margin(Row, NewScore)
        if Name in Cows and NewScore is not None and Row[Cows.index(Name)] < NewScore:
            Name = "new"
        if Name in Cows and Margin < Tau:
            Flags[T].add("low margin")
            Name = "?"
        Rows_.append({"TrackId": int(T), "Name": Name, "Score": round(float(Row.max()), 4), "Margin": round(float(Margin), 4),
                      "Crops": len(Group), "FirstSeconds": float(Spans.loc[T, "min"]), "LastSeconds": float(Spans.loc[T, "max"]),
                      "Flags": "; ".join(sorted(Flags[T]))})
    Names = pd.DataFrame(Rows_, columns=["TrackId", "Name", "Score", "Margin", "Crops", "FirstSeconds", "LastSeconds", "Flags"])
    CropNames = Query.merge(Names[["TrackId", "Name", "Flags"]], on="TrackId", how="left")
    return Names, CropNames, pd.DataFrame(Records)
