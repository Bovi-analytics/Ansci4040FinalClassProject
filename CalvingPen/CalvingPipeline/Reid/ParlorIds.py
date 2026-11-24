"""Real cow IDs from the milking parlor, without labelling by hand.

The BoviSync parlor report (``BoviSyncRecords/Nov17Parlor.xlsx``) lists every
milking: cow (``EART``), attach time (``PaAttachdtm``) and stall (``PaStall``,
1-28 one side, 29-56 the other). A **turn** is one side filling up: its cows
share one attach minute. On a day with 3 milkings every cow is in 3 turns,
with different neighbours each time. So:

1. ``ReadParlorReport``: the report, with ``Session`` (milking), ``Side`` and
   ``Turn`` per row (rows without a stall are left out: not milked).
2. The video's clock (``PartOffsets``): the entry-parlor camera records all
   the time and cuts a file every 8 h (00:00, 08:00, 16:00), so each part's
   start is its place in the day. ``AlignClock`` (busy minutes = milkings)
   only works for cameras that save a picture only when something moves.
3. ``SessionWindows``: the video stretch of each milking, to track only those.
4. ``TrackCandidates``: per track (TrackAndCrop, run on those windows), the
   cows of every turn whose cows could be in the parlor at the track's time,
   on the track's side of the picture.
5. ``IdentifyCows``: per cow, one track in each milking whose candidates hold
   that cow, chosen so the tracks look most alike (8-band DINOv3 fingerprints).
   A cow is named only when that best choice clearly beats the next one
   (``MinMargin``) and her tracks really look alike (``MinScore``); a track
   gets at most one cow. Which side of the picture is stalls 1-28 is not
   known, so both ways are tried and the one with more alike tracks is kept.

The result, ``ParlorIds.csv`` (TrackId, Session, CowId, Score, Margin), is the
labelled pool: those tracks' crops are the anchors with a real farm ID.
``MatchCalving`` then names calving-pen tracks against it.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .Identity import CropScores, Parts3


# ---------------------------------------------------------------- the report

def ReadParlorReport(Path_, StallsPerSide=28, SessionGapMinutes=90, TurnGapMinutes=3) -> pd.DataFrame:
    """One row per milking with a stall: CowId, Attach (time of day), Stall,
    Side (A = stalls 1-28, B = 29-56), Session (1, 2, 3 by time), Turn."""
    Path_ = Path(Path_)
    Read = pd.read_excel if Path_.suffix.lower() in (".xlsx", ".xls") else pd.read_csv
    Raw = Read(Path_, header=None, dtype=str)
    Header = next(I for I, Row in Raw.iterrows() if "EART" in [str(V).strip() for V in Row.values])
    Table = Raw.iloc[Header + 1:].copy()
    Table.columns = [str(V).strip() for V in Raw.iloc[Header].values]
    Out = pd.DataFrame({
        "CowId": pd.to_numeric(Table["EART"], errors="coerce"),
        "Attach": pd.to_datetime(Table["PaAttachdtm"], errors="coerce"),
        "Stall": pd.to_numeric(Table["PaStall"], errors="coerce"),
        "Pen": Table.get("PaPen"),
        "DurationSeconds": pd.to_numeric(Table.get("PaDur"), errors="coerce"),
    }).dropna(subset=["CowId", "Attach", "Stall"])
    Out = Out.astype({"CowId": int, "Stall": int}).sort_values(["Attach", "Stall"]).reset_index(drop=True)
    Out["Side"] = np.where(Out.Stall <= StallsPerSide, "A", "B")
    Minutes = (Out.Attach - Out.Attach.dt.normalize()).dt.total_seconds() / 60
    Out["ClockMinute"] = Minutes
    Out["Session"] = (Minutes.diff().fillna(0) > SessionGapMinutes).cumsum() + 1
    Turns = []
    for (_, _), G in Out.groupby(["Session", "Side"], sort=False):   # a straggler a few minutes later joins its turn
        New = G.ClockMinute.diff().fillna(np.inf) > TurnGapMinutes
        Turns.append(pd.Series(New.cumsum().to_numpy(), index=G.index))
    Number = pd.concat(Turns)
    Out["Turn"] = [f"S{S}{Side}{N}" for S, Side, N in zip(Out.Session, Out.Side, Number.loc[Out.index])]
    Out["TurnMinute"] = Out.groupby("Turn").ClockMinute.transform("min")
    return Out


def ReadCalvings(Path_) -> pd.DataFrame:
    """BoviSync event report (``Nov11To15Calvings.xlsx``): CowId, Date, Pen,
    PreviousPen, Lact of every Freshen (calving) event."""
    Path_ = Path(Path_)
    Read = pd.read_excel if Path_.suffix.lower() in (".xlsx", ".xls") else pd.read_csv
    Raw = Read(Path_, header=None, dtype=str)
    Header = next(I for I, Row in Raw.iterrows() if "CowId" in [str(V).strip() for V in Row.values])
    Table = Raw.iloc[Header + 1:].copy()
    Table.columns = [str(V).strip() for V in Raw.iloc[Header].values]
    Table = Table[Table.get("EvType", "Freshen").astype(str).str.lower().str.startswith("fresh")]
    return pd.DataFrame({"CowId": pd.to_numeric(Table.CowId, errors="coerce"),
                         "Date": pd.to_datetime(Table.EvDate, errors="coerce").dt.normalize(),
                         "Pen": Table.get("Pen"), "PreviousPen": Table.get("PreviousPen"),
                         "Lact": pd.to_numeric(Table.get("Lact"), errors="coerce")}).dropna(subset=["CowId", "Date"]).astype(
        {"CowId": int}).reset_index(drop=True)


def SessionSpans(Report, BeforeMinutes=12, AfterMinutes=8) -> pd.DataFrame:
    """Clock minutes of each milking: the first turn's cows walk in about
    ``BeforeMinutes`` before the first attach; the last leave after the last."""
    S = Report.groupby("Session").agg(First=("ClockMinute", "min"), Last=("ClockMinute", "max"), Cows=("CowId", "nunique"))
    S["Start"], S["End"] = S.First - BeforeMinutes, S.Last + AfterMinutes
    return S.reset_index()


# ---------------------------------------------------------------- the clock

def AlignClock(Activity, Report, MinOffset=0, MaxOffset=24 * 60, **Spans) -> dict:
    """``OffsetMinutes``: clock minute = video minute + offset. Tries every
    whole minute and keeps the one where the busy minutes of the video fall
    inside the milkings. ``Activity``: ``AviActivity`` (Minute, RealFrames).
    ``Confidence``: the share of the video's real frames inside the milkings
    at that offset (the best other offset in brackets, to see it is clear)."""
    Counts = Activity.RealFrames.to_numpy(float)
    Busy = np.log1p(Counts)
    Spans_ = SessionSpans(Report, **Spans)
    Scores = []
    for Offset in range(int(MinOffset), int(MaxOffset) + 1):
        Clock = np.arange(len(Busy)) + Offset
        Inside = np.zeros(len(Busy), bool)
        for S in Spans_.itertuples():
            Inside |= (Clock >= S.Start) & (Clock <= S.End)
        Scores.append((Busy[Inside].sum() - Busy[~Inside].mean() * Inside.sum() if Inside.any() else -np.inf, Offset,
                       Counts[Inside].sum() / max(Counts.sum(), 1)))
    Scores.sort(reverse=True)
    Best = Scores[0]
    Far = next((S for S in Scores[1:] if abs(S[1] - Best[1]) > 30), Scores[-1])   # best offset more than 30 min away
    return {"OffsetMinutes": Best[1], "ShareInside": round(Best[2], 3), "NextBestOffset": Far[1],
            "NextBestShare": round(Far[2], 3)}


def PartOffsets(VideoMinutes, Starts=None, DayStart="00:00") -> list:
    """Clock minute at which each part of a day's recording starts. The entry
    parlor camera records all the time (900 pictures every minute) and cuts a
    new file every 8 h, so by default the parts follow each other from
    ``DayStart``. ``Starts``: one "HH:MM" per part to set them by hand."""
    def Minute(Text):
        H, M = map(int, str(Text).split(":")[:2])
        return H * 60 + M
    if Starts:
        return [Minute(S) for S in Starts]
    Out, Now = [], Minute(DayStart)
    for M in VideoMinutes:
        Out.append(Now)
        Now += M
    return Out


def RecordedMinutes(Activity) -> float:
    """Length of a recording in minutes from ``AviActivity`` (the last minute
    counted by the share of it that holds pictures)."""
    Counts = Activity.RealFrames.to_numpy(float)
    if len(Counts) < 2:
        return float(len(Counts))
    return len(Counts) - 1 + min(1.0, Counts[-1] / max(np.median(Counts), 1))


def SteadyRecording(Activity, MaxSpread=0.05) -> bool:
    """True when the real pictures per minute hardly change (the camera
    records all the time): then ``AlignClock`` cannot find the milkings."""
    Counts = Activity.RealFrames.to_numpy(float)[:-1] if len(Activity) > 1 else Activity.RealFrames.to_numpy(float)
    return bool(len(Counts) and Counts.std() <= MaxSpread * max(Counts.mean(), 1))


def MilkingFrames(VideoPath, Windows, OutPath, Width=640, NativeFps=None):
    """One picture from the middle of each milking window, side by side, to
    check by eye that the clock is right (cows in the stalls)."""
    import cv2

    from .Frames import SampleFrames

    Tiles = []
    for W in Windows:
        Middle = (W["StartSeconds"] + W["EndSeconds"]) / 2
        for _, T, Frame in SampleFrames(VideoPath, 1, Middle, Middle + 2, NativeFps):
            Scale = Width / Frame.shape[1]
            Tile = cv2.resize(Frame, None, fx=Scale, fy=Scale)
            cv2.putText(Tile, f"milking {W['Session']}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            Tiles.append(Tile)
            break
    if Tiles:
        H = max(T.shape[0] for T in Tiles)
        cv2.imwrite(str(OutPath), np.hstack([cv2.copyMakeBorder(T, 0, H - T.shape[0], 0, 0, cv2.BORDER_CONSTANT) for T in Tiles]))
    return OutPath if Tiles else None


def SessionWindows(Report, OffsetMinutes, VideoMinutes, **Spans) -> pd.DataFrame:
    """Per milking inside this video: StartSeconds, EndSeconds (video time)."""
    S = SessionSpans(Report, **Spans)
    S["StartSeconds"] = ((S.Start - OffsetMinutes) * 60).clip(lower=0)
    S["EndSeconds"] = ((S.End - OffsetMinutes) * 60).clip(upper=VideoMinutes * 60)
    return S[S.EndSeconds > S.StartSeconds + 60].reset_index(drop=True)


# ---------------------------------------------------------------- tracks -> candidate cows

def TrackCandidates(Crops, Report, OffsetMinutes, Width, SideOnLeft="A", EnterMinutes=8, StayMinutes=12,
                    Middle=0.5) -> pd.DataFrame:
    """Per track: its Session, Side (left or right half of the picture, as
    stall side ``SideOnLeft``) and ``Candidates``: the cows of the turns of
    that side whose time in the parlor (``EnterMinutes`` before attach to
    ``StayMinutes`` after) holds the track's middle time."""
    Spans = SessionSpans(Report)
    Turns = Report.groupby("Turn").agg(Side=("Side", "first"), Session=("Session", "first"),
                                       Minute=("TurnMinute", "first"), Cows=("CowId", lambda C: set(C)))
    Rows = []
    for TrackId, T in Crops.groupby("TrackId"):
        Clock = OffsetMinutes + T.TimeSeconds.median() / 60
        X = ((T.X1 + T.X2) / 2).median() / Width
        Side = SideOnLeft if X < Middle else ("B" if SideOnLeft == "A" else "A")
        Live = Turns[(Turns.Side == Side) & (Turns.Minute - EnterMinutes <= Clock) & (Clock <= Turns.Minute + StayMinutes)]
        Session = Spans[(Spans.Start <= Clock) & (Clock <= Spans.End)].Session
        Rows.append({"TrackId": TrackId, "Session": int(Session.iloc[0]) if len(Session) else None, "Side": Side,
                     "ClockMinute": round(Clock, 1), "Crops": len(T),
                     "Candidates": set().union(*Live.Cows) if len(Live) else set()})
    return pd.DataFrame(Rows)


# ---------------------------------------------------------------- naming

def TrackVectors(Crops, Bands):
    """One fingerprint per track: the mean of its crops' 3 body thirds, both
    orientations (so a cow standing the other way still matches)."""
    A, B = Parts3(Bands)
    Ids = Crops.TrackId.to_numpy()
    Out = {}
    for T in np.unique(Ids):
        Sel = Ids == T
        V = np.concatenate([A[Sel].mean(0), B[Sel].mean(0)])
        Out[T] = V / (np.linalg.norm(V) + 1e-8)
    return Out


def _BestPick(Lists, Vectors):
    """Best track per list (one per milking) by mean pairwise similarity, and
    the best pick sharing no track with it. Vectorised over all picks."""
    M = [np.stack([Vectors[T] for T in L]) for L in Lists]
    K = len(M)
    Total = np.zeros([len(L) for L in Lists], np.float32)
    for I in range(K):
        for J in range(I + 1, K):
            Shape = [1] * K
            Shape[I], Shape[J] = len(Lists[I]), len(Lists[J])
            Total = Total + (M[I] @ M[J].T).reshape(Shape)
    Total /= K * (K - 1) / 2
    Index = np.unravel_index(int(np.argmax(Total)), Total.shape)
    Best = float(Total[Index])
    Other = Total.copy()
    for Axis, I in enumerate(Index):                 # rivals may not reuse any of the winner's tracks
        Slicer = [slice(None)] * K
        Slicer[Axis] = I
        Other[tuple(Slicer)] = -np.inf
    Rival = float(Other.max()) if np.isfinite(Other.max()) else -1.0
    return [L[I] for L, I in zip(Lists, Index)], Best, Rival


def IdentifyCows(Candidates, Vectors, MinSessions=2, MinScore=0.75, MinMargin=0.03, MaxTracksPerSession=60) -> pd.DataFrame:
    """Per cow, one track per milking (at least ``MinSessions``) whose
    candidates hold her and that look most alike (mean pairwise similarity).
    Greedy: the clearest cow first (largest margin over the best other
    choice), then her tracks are taken and the rest is redone. Returns the
    named tracks: TrackId, Session, CowId, Score, Margin."""
    C = Candidates.dropna(subset=["Session"])
    C = C[C.TrackId.isin(Vectors)]
    Cows = sorted(set().union(*C.Candidates)) if len(C) else []
    Sessions = [G for _, G in C.groupby("Session")]
    Taken, Named = set(), []
    while Cows:
        Best = None
        for Cow in Cows:
            Lists = [G.TrackId[[Cow in S and T not in Taken for T, S in zip(G.TrackId, G.Candidates)]].tolist()[:MaxTracksPerSession]
                     for G in Sessions]
            Lists = [L for L in Lists if L]
            if len(Lists) < MinSessions:
                continue
            Pick, Score, Rival = _BestPick(Lists, Vectors)
            Margin = Score - Rival
            if Score >= MinScore and Margin >= MinMargin and (Best is None or Margin > Best[2]):
                Best = (Cow, Pick, Margin, Score)
        if Best is None:
            break
        Cow, Pick, Margin, Score = Best
        for T in Pick:
            Named.append({"TrackId": T, "Session": int(C.Session[C.TrackId == T].iloc[0]), "CowId": Cow,
                          "Score": round(Score, 3), "Margin": round(Margin, 3)})
            Taken.add(T)
        Cows.remove(Cow)
    return pd.DataFrame(Named, columns=["TrackId", "Session", "CowId", "Score", "Margin"])


def NameParlor(Crops, Bands, Report, OffsetMinutes, Width, OutputDir=None, **Settings) -> pd.DataFrame:
    """``IdentifyCows`` with both side layouts; keeps the one naming more cows
    (then the higher mean score). Writes ParlorIds.csv and ParlorCandidates.csv."""
    Vectors = TrackVectors(Crops, Bands)
    Tried = []
    for Left in ("A", "B"):
        Cand = TrackCandidates(Crops, Report, OffsetMinutes, Width, SideOnLeft=Left)
        Ids = IdentifyCows(Cand, Vectors, **Settings)
        Tried.append((Ids.CowId.nunique(), Ids.Score.mean() if len(Ids) else 0, Left, Ids, Cand))
        print(f"Stalls 1-28 on the {'left' if Left == 'A' else 'right'}: {Ids.CowId.nunique()} cows named")
    Tried.sort(key=lambda X: (X[0], X[1]), reverse=True)
    _, _, Left, Ids, Cand = Tried[0]
    Ids["SideOnLeft"] = Left
    if OutputDir:
        OutputDir = Path(OutputDir)
        OutputDir.mkdir(parents=True, exist_ok=True)
        Ids.to_csv(OutputDir / "ParlorIds.csv", index=False)
        Cand.assign(Candidates=Cand.Candidates.map(lambda S: " ".join(map(str, sorted(S))))).to_csv(
            OutputDir / "ParlorCandidates.csv", index=False)
    return Ids


# ---------------------------------------------------------------- calving pen -> parlor

def MatchCalving(Crops, Bands, PoolCrops, PoolBands, Expected=None, TopK=5) -> pd.DataFrame:
    """Per calving-pen track: the parlor cow it looks most like (``CropScores``
    averaged over its crops), the runner-up, the margin, and whether the best
    cow is in ``Expected`` (e.g. the cows that calved that day: a check, since
    the pool also holds many cows that did not)."""
    Cows = sorted(PoolCrops.CowId.unique())
    Scores = CropScores(Bands, PoolBands, PoolCrops.CowId.to_numpy(), Cows, TopK=min(TopK, PoolCrops.CowId.value_counts().min()))
    Rows = []
    for TrackId, Sel in Crops.reset_index(drop=True).groupby("TrackId").groups.items():
        Mean = np.nanmean(Scores[list(Sel)], 0)
        Order = np.argsort(-np.nan_to_num(Mean, nan=-9))
        Row = {"TrackId": TrackId, "Crops": len(Sel), "BestCow": Cows[Order[0]], "Score": round(float(Mean[Order[0]]), 3),
               "SecondCow": Cows[Order[1]] if len(Cows) > 1 else None,
               "Margin": round(float(Mean[Order[0]] - Mean[Order[1]]), 3) if len(Cows) > 1 else None,
               "Expected": (Cows[Order[0]] in Expected) if Expected is not None else None}
        if Expected is not None:   # the best of the expected cows, and where she ranks among all pool cows
            Ranked = [Cows[I] for I in Order]
            Best = next((C for C in Ranked if C in Expected), None)
            Row.update(BestExpected=Best, ExpectedRank=Ranked.index(Best) + 1 if Best is not None else None)
        Rows.append(Row)
    return pd.DataFrame(Rows).sort_values("Crops", ascending=False).reset_index(drop=True)


def ActiveWindows(Activity, Count=2, Minutes=15) -> list:
    """``Count`` non-overlapping ``Minutes``-long stretches with the most real
    frames (a cow moving), as (StartSeconds, EndSeconds), in time order."""
    Counts = Activity.RealFrames.to_numpy(float)
    if len(Counts) <= Minutes:
        return [(0, len(Counts) * 60)]
    Sums = np.convolve(Counts, np.ones(Minutes), "valid")
    Out = []
    for _ in range(Count):
        I = int(np.argmax(Sums))
        if Sums[I] <= 0:
            break
        Out.append((I * 60, (I + Minutes) * 60))
        Sums[max(0, I - Minutes + 1):I + Minutes] = -1
    return sorted(Out)


# ---------------------------------------------------------------- several runs together

def LoadMany(Dirs, ClockOffsetMinutes=None, GateArea=10000, PenAreas=None, **Gates):
    """Crops and fingerprints of several TrackAndCrop folders as one table.
    ``TrackId`` becomes ``<folder number> * 1,000,000 + TrackId`` (unique),
    ``Folder`` says where a crop came from, and with ``ClockOffsetMinutes``
    (one per folder) ``TimeSeconds`` becomes seconds of the day. ``GateArea``
    is for a 1920 x 1080 picture and is scaled to each folder's resolution.
    ``PenAreas``: one polygon (or None) per folder; None = no pen filter."""
    import cv2
    from .Validate import LoadVideo

    AllCrops, AllBands = [], []
    for N, Dir in enumerate(map(Path, Dirs)):
        if not (Dir / "Bands.npz").exists():
            print(f"{Dir.name}: no Bands.npz yet, left out")
            continue
        Frame = cv2.imread(str(Dir / "FirstFrame.jpg"))
        Scale = (Frame.shape[0] * Frame.shape[1]) / (1920 * 1080) if Frame is not None else 1.0
        Crops, Bands, _ = LoadVideo(Dir, PenArea=PenAreas[N] if PenAreas else None, GateArea=GateArea * Scale, **Gates)
        Crops = Crops.assign(Folder=Dir.name, FullPath=[str(Dir / P) for P in Crops.CropPath],
                             LocalTrackId=Crops.TrackId, TrackId=N * 1_000_000 + Crops.TrackId,
                             Width=Frame.shape[1] if Frame is not None else 1920)
        if ClockOffsetMinutes is not None:
            Crops["TimeSeconds"] = Crops.TimeSeconds + ClockOffsetMinutes[N] * 60
        AllCrops.append(Crops)
        AllBands.append(Bands)
    if not AllCrops:
        raise ValueError("No folder with Bands.npz")
    return pd.concat(AllCrops, ignore_index=True), np.concatenate(AllBands)


def CowSheets(Crops, Ids, OutputDir, PerTrack=6, TileSize=120) -> Path:
    """``<OutputDir>/Cows/<CowId>.jpg``: one row per milking (track) of a
    named cow, to check by eye that the rows show the same cow."""
    import cv2
    from .Labels import _Sheet, _Tile

    Out = Path(OutputDir) / "Cows"
    Out.mkdir(parents=True, exist_ok=True)
    for Old in Out.glob("*.jpg"):
        Old.unlink()
    for Cow, G in Ids.groupby("CowId"):
        Tiles = []
        for T in G.sort_values("Session").TrackId:
            C = Crops[Crops.TrackId == T].sort_values("TimeSeconds")
            Pick = C.iloc[np.unique(np.linspace(0, len(C) - 1, min(PerTrack, len(C))).round().astype(int))]
            Row = [_Tile(Path(P), TileSize, f"S{int(G.Session[G.TrackId == T].iloc[0])}") for P in Pick.FullPath]
            Tiles += Row + [np.full((TileSize, TileSize, 3), 30, np.uint8)] * (PerTrack - len(Row))
        cv2.imwrite(str(Out / f"{Cow}.jpg"), _Sheet(Tiles, PerTrack, TileSize))
    return Out


# ---------------------------------------------------------------- one milking: calving cows -> their parlor turn

def TurnCows(Report, ClockMinute, Side=None, EnterMinutes=8, StayMinutes=12) -> set:
    """Cows of every turn that could be in the parlor at ``ClockMinute``
    (only stall side ``Side`` "A"/"B", or both)."""
    Live = Report[(Report.TurnMinute - EnterMinutes <= ClockMinute) & (ClockMinute <= Report.TurnMinute + StayMinutes)]
    if Side is not None:
        Live = Live[Live.Side == Side]
    return set(Live.CowId)


def RankCalvedCows(Crops, Bands, ParlorCrops, ParlorBands, Report, Calved, SideOnLeft=None, MinCrops=3, Middle=0.5,
                   **Turn) -> pd.DataFrame:
    """With ONE milking: per calving-pen track, every cow in ``Calved`` (the
    cows that calved around the video's day) gets the similarity of the
    most alike parlor track that could be her: a track on her stall side of
    the picture whose time falls in her turn. ``SideOnLeft``: the stall side
    ("A" = stalls 1-28) on the left half of the picture; None = try both and
    keep the layout whose best cows are clearer (higher mean margin), see
    ``Layout``. Returns per track: the best cow, its score, the runner-up,
    the margin, and how many calved cows were milked then (chance = 1 / that).
    ``ParlorCrops.TimeSeconds`` must be seconds of the day."""
    if SideOnLeft is None:
        Tried = [RankCalvedCows(Crops, Bands, ParlorCrops, ParlorBands, Report, Calved, S, MinCrops, Middle, **Turn)
                 for S in ("A", "B")]
        Tried = [T for T in Tried if len(T)]
        return max(Tried, key=lambda T: T.Margin.fillna(0).mean()) if Tried else pd.DataFrame()
    P = TrackVectors(ParlorCrops, ParlorBands)
    PIds = np.array(list(P))
    PMat = np.stack([P[T] for T in PIds])
    G = ParlorCrops.groupby("TrackId")
    Minute = G.TimeSeconds.median() / 60
    X = ((ParlorCrops.X1 + ParlorCrops.X2) / 2 / ParlorCrops.get("Width", 1920)).groupby(ParlorCrops.TrackId).median()
    Other = "B" if SideOnLeft == "A" else "A"
    Side = {T: SideOnLeft if X[T] < Middle else Other for T in PIds}
    Holds = {Cow: np.array([Cow in TurnCows(Report, Minute[T], Side[T], **Turn) for T in PIds]) for Cow in Calved}
    Present = [Cow for Cow in Calved if Holds[Cow].any()]
    Q = TrackVectors(Crops, Bands)
    Counts = Crops.TrackId.value_counts()
    Rows = []
    for T, V in Q.items():
        if Counts[T] < MinCrops or not Present:
            continue
        Sim = PMat @ V
        Score = {Cow: float(Sim[Holds[Cow]].max()) for Cow in Present}
        Order = sorted(Score, key=Score.get, reverse=True)
        Rows.append({"TrackId": T, "Crops": int(Counts[T]), "BestCow": Order[0], "Score": round(Score[Order[0]], 3),
                     "SecondCow": Order[1] if len(Order) > 1 else None,
                     "Margin": round(Score[Order[0]] - Score[Order[1]], 3) if len(Order) > 1 else None,
                     "CalvedInParlor": len(Present), "Layout": f"stalls 1-28 {'left' if SideOnLeft == 'A' else 'right'}"})
    return pd.DataFrame(Rows).sort_values("Crops", ascending=False).reset_index(drop=True) if Rows else pd.DataFrame()


# ---------------------------------------------------------------- crops made earlier by the step pipeline

def ImportOldCrops(StepDir, OutputDir, FrameSize=(1080, 1920)) -> pd.DataFrame:
    """Use the crops of the earlier step pipeline (``<video>/Step04bCropFull``:
    Crops.csv with TrackId, TimeSeconds, CropPath, box, MaskFraction) without
    tracking again: writes a Crops.csv in this pipeline's form in
    ``OutputDir`` that points at the existing crop images (full paths; nothing
    is copied), plus Tracks.csv and a blank FirstFrame.jpg of the video's
    size (so the gates scale). ``EmbedBands`` and ``LoadVideo`` then work on
    ``OutputDir``. Skipped when done for the same Crops.csv."""
    import cv2

    from .Segments import CropColumns

    StepDir, OutputDir = Path(StepDir), Path(OutputDir)
    Source = StepDir / "Crops.csv"
    OutputDir.mkdir(parents=True, exist_ok=True)
    Stamp = f"{Source}:{Source.stat().st_size}"
    Done = OutputDir / "Source.txt"
    if Done.exists() and Done.read_text() == Stamp and (OutputDir / "Crops.csv").exists():
        print(f"{OutputDir.name}: already imported")
        return pd.read_csv(OutputDir / "Crops.csv")
    Old = pd.read_csv(Source)
    W, H = (Old.X2 - Old.X1).clip(lower=1), (Old.Y2 - Old.Y1).clip(lower=1)
    Area = (W * H * Old.get("MaskFraction", pd.Series(0.5, index=Old.index))).round().astype(int)
    Table = pd.DataFrame({
        "TrackId": Old.TrackId, "Tick": Old.get("Frame", 0), "TimeSeconds": Old.TimeSeconds,
        "CropPath": [str(StepDir / P) for P in Old.CropPath], "X1": Old.X1, "Y1": Old.Y1, "X2": Old.X2, "Y2": Old.Y2,
        "Confidence": Old.get("Confidence", 1.0), "MaskArea": Area, "NeighbourIos": 0.0, "Elongation": np.nan,
        "BoxAspect": (W / H).round(3), "ColouredFraction": 0.0, "FrameSaturation": 40.0, "Night": False})[CropColumns]
    Table = Table[[Path(P).exists() for P in Table.CropPath]].reset_index(drop=True)
    Table.to_csv(OutputDir / "Crops.csv", index=False)
    Table[["Tick", "TimeSeconds", "TrackId", "X1", "Y1", "X2", "Y2", "Confidence"]].to_csv(OutputDir / "Tracks.csv", index=False)
    cv2.imwrite(str(OutputDir / "FirstFrame.jpg"), np.zeros((*FrameSize, 3), np.uint8))
    Done.write_text(Stamp)
    print(f"{OutputDir.name}: {len(Table)} crops of {Table.TrackId.nunique()} tracks from {StepDir.parent.name}")
    return Table
