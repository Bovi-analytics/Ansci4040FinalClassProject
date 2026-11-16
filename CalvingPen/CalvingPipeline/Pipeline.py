"""Run the whole pipeline (steps 01-06) for one or many videos with one call.

The step notebooks run one step at a time, to check each result by eye.
Once the settings are right, ``RunVideo`` runs steps 01-05 for one video in
order, and ``RunVideos`` does that for a list of videos (one cow or several)
and then pools them all (step 06). They call exactly the same functions as
the notebooks and write to the same folders, so the results are identical:

    CalvingPenPipelineOutputs/<VideoStem>/Step02bDetectFull/ ... Step05bEmbedFull/
    CalvingPenPipelineOutputs/Pool/

``Test=True`` runs the "a" versions instead (a short window, folders
``Step02aDetectTest`` ... and ``PoolTest``). Every step overwrites its own
folder; only step 03 continues a run that stopped part-way.
"""

import time
import traceback
from pathlib import Path

from .Paths import FindVideo, OutputRoot as DefaultOutputRoot
from .VideoTools import PrepareClips

StepNames = {1: "PrepareClip", 2: "Detect", 3: "Track", 4: "Crop", 5: "Embed", 6: "Pool"}


def StepFolders(Test=False) -> dict:
    """Output folder per step, the same names the notebooks use."""
    Letter, Kind = ("a", "Test") if Test else ("b", "Full")
    return {Step: f"Step0{Step}{Letter}{StepNames[Step]}{Kind}" for Step in (2, 3, 4, 5)} | \
           {6: "PoolTest" if Test else "Pool"}


def RunVideo(VideoStem=None, VideoFile=None, ClipStart=None, ClipEnd=None, FromStep=1, ToStep=5, Test=False,
             TestStart=1080, TestSeconds=240, OutputRoot=None, HfToken=None,
             TargetFps=2, Confidence=0.25, SeedConfidence=0.4, ChunkSeconds=120,
             MinFrames=10, MinConfidence=0.4, CropsPerTrack=None, CropEverySeconds=15,
             GroupSeconds=None, Preview=True, Half=False) -> dict:
    """Steps ``FromStep``..``ToStep`` (1-5) for one video.

    - ``VideoFile``: the raw video's file name, e.g. "2026-09-26-CalvingPen3(1).avi" (found with
      ``FindVideo``); only needed for step 01, together with ``ClipStart`` / ``ClipEnd``
    - ``VideoStem``: the output folder name; default = the file name without extension
    - ``Test``: run on ``TestStart`` .. ``TestStart + TestSeconds`` of the clip only
    Returns each step's numbers; raises if a step fails (later steps need its output).
    """
    from .Crop import CropCows
    from .Detect import DetectCows
    from .Embed import EmbedCrops
    from .Track import TrackCows

    if VideoStem is None:
        if VideoFile is None:
            raise ValueError("Give VideoFile (the raw video's name) or VideoStem (its output folder)")
        VideoStem = Path(VideoFile).stem
    VideoDir = Path(OutputRoot or DefaultOutputRoot) / VideoStem
    ClipPath = VideoDir / "ReferenceClip.mp4"
    Folder = {Step: VideoDir / Name for Step, Name in StepFolders(Test).items() if Step <= 5}
    Start, End = (TestStart, TestStart + TestSeconds) if Test else (0, None)
    CropsPerTrack = CropsPerTrack or (8 if Test else 24)
    GroupSeconds = GroupSeconds or (60 if Test else 300)
    Results, Started = {}, time.time()

    for Step in range(FromStep, ToStep + 1):
        print(f"\n=== {VideoStem}: step 0{Step} {StepNames[Step]} ===")
        StepStarted = time.time()
        if Step == 1:
            if VideoFile is None:
                raise ValueError("Step 01 needs VideoFile (or start at FromStep=2 if ReferenceClip.mp4 exists)")
            Summary = PrepareClips(FindVideo(VideoFile), VideoDir, ClipStart or 0, ClipEnd)
            Results[Step] = {"ClipStart": Summary["ClipStart"], "ClipEnd": Summary["ClipEnd"]}
        elif Step == 2:
            Results[Step] = DetectCows(ClipPath, Folder[2], Start, End, TargetFps=TargetFps, Confidence=Confidence,
                                       Preview=Preview and Test)
        elif Step == 3:
            Results[Step] = TrackCows(ClipPath, Folder[2], Folder[3], ChunkSeconds=ChunkSeconds,
                                      SeedConfidence=SeedConfidence, Preview=Preview, Half=Half,
                                      CompareTracksPath=VideoDir / "BaselineByteTrack" / "Tracks.csv")
        elif Step == 4:
            Results[Step] = CropCows(ClipPath, Folder[3] / "Tracks.csv", Folder[4], MinFrames=MinFrames,
                                     MinConfidence=MinConfidence, CropsPerTrack=CropsPerTrack,
                                     CropEverySeconds=CropEverySeconds)
        elif Step == 5:
            Results[Step] = EmbedCrops(Folder[4], Folder[5], VideoStem, Token=HfToken, GroupSeconds=GroupSeconds)
        print(f"step 0{Step} done in {(time.time() - StepStarted) / 60:.1f} min")
    print(f"\n{VideoStem}: steps 0{FromStep}-0{ToStep} done in {(time.time() - Started) / 60:.1f} min")
    return Results


def RunVideos(Videos, Test=False, Pool=True, OutputRoot=None, HfToken=None, ExcludeSeconds=None, **Settings) -> dict:
    """``RunVideo`` for every entry of ``Videos`` (dicts with VideoFile and/or
    VideoStem, ClipStart / ClipEnd for step 01, FromStep / ToStep), then step 06 over
    all video folders. A video that fails is reported and skipped; the
    others still run."""
    from .Pool import BuildPool

    OutputRoot = Path(OutputRoot or DefaultOutputRoot)
    Results, Failed = {}, {}
    for Video in Videos:
        Video = dict(Video)
        Stem = Video.pop("VideoStem", None) or Path(Video.get("VideoFile", "")).stem
        try:
            Results[Stem] = RunVideo(Stem, Test=Test, OutputRoot=OutputRoot, HfToken=HfToken, **Video, **Settings)
        except Exception as Error:   # keep going with the other videos
            traceback.print_exc()
            Failed[Stem] = f"{type(Error).__name__}: {Error}"
    if Pool:
        print("\n=== step 06 Pool ===")
        Folders = StepFolders(Test)
        Results["Pool"] = BuildPool(OutputRoot, OutputRoot / Folders[6], EmbeddingFolder=Folders[5],
                                    ExcludeSeconds=ExcludeSeconds or (60 if Test else 300))
    if Failed:
        print("\nFAILED:", *[f"  {S}: {E}" for S, E in Failed.items()], sep="\n")
    Results["Failed"] = Failed
    return Results


def VideoStatus(OutputRoot=None, Test=False):
    """One row per video folder: which steps are done, which tracker the crops
    came from, whether the fingerprints are up to date and with which model,
    and whether the video is in the pool. Shows at a glance that only SAM
    2.1 outputs are used."""
    import json

    import pandas as pd

    from .Embed import DinoV3, CropSource, EmbedUpToDate

    OutputRoot = Path(OutputRoot or DefaultOutputRoot)
    F = StepFolders(Test)
    PoolIndex = OutputRoot / F[6] / "PoolIndex.csv"
    InPool = set(pd.read_csv(PoolIndex).VideoStem) if PoolIndex.exists() else set()
    Rows = []
    for VideoDir in sorted(P for P in OutputRoot.iterdir() if P.is_dir() and P.name not in ("Pool", "PoolTest", "Old")):
        Tracks = VideoDir / F[3] / "Tracks.csv"
        Crops = VideoDir / F[4] / "Crops.csv"
        Model = VideoDir / F[5] / "Model.txt"
        Rows.append({
            "VideoStem": VideoDir.name,
            "Clip": (VideoDir / "ReferenceClip.mp4").exists(),
            "Detect02": (VideoDir / F[2] / "Detections.csv").exists(),
            "SamTracks03": (VideoDir / F[3] / "Chunks" / "Settings.json").exists() and Tracks.exists(),
            "Tracks": int(pd.read_csv(Tracks, usecols=["TrackId"]).TrackId.nunique()) if Tracks.exists() else None,
            "Crops04": int(len(pd.read_csv(Crops))) if Crops.exists() else None,
            "CropsFrom": CropSource(VideoDir, F[4]) if Crops.exists() else "",
            "Embed05UpToDate": EmbedUpToDate(VideoDir, F[4], F[5]),
            "Model05": Model.read_text().strip().split("/")[-1] if Model.exists() else "",
            "InPool06": VideoDir.name in InPool,
            "OldFolders": sorted(P.name for P in VideoDir.iterdir() if P.is_dir() and P.name in (
                "Step02aTest", "Step02bFull", "Step02cSam2", "Step02dSam2Full", "Step03aTest", "Step03bFull",
                "Step04aTest", "Step04bFull")) or "",
        })
    Status = pd.DataFrame(Rows)
    if len(Status):
        Ready = Status.Crops04.notna() & Status.CropsFrom.str.startswith("SAM")
        print(f"{len(Status)} video folders; {int(Ready.sum())} with SAM crops; "
              f"{int(Status.Embed05UpToDate.sum())} with up-to-date {DinoV3.split('/')[-1]} fingerprints; "
              f"{int(Status.InPool06.sum())} in {F[6]}/")
    return Status
