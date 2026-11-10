"""Where the videos and the pipeline outputs live on the group's Shared Drive
(as mounted in Colab). Change it here and every notebook follows.

Note the space at the end of "CalvingPenVideos ".
The group folder was renamed from "ANSC 4040 Group Project" in Nov 2026; if a
folder is renamed again, only these three lines need to change.

``ListVideos()`` shows the calving videos that are there, ``FindVideo(Name)``
finds one by its file name (also if it was moved to another folder of the
group drive), so notebooks only need the file name.
"""

from pathlib import Path

GroupDir = Path("/content/drive/Shareddrives/BeastsAndBytes/ANSC4040GroupProject")
RawVideoDir = GroupDir / "CalvingPenVideos "
OutputRoot = GroupDir / "CalvingPenPipelineOutputs"   # one subfolder per video, then one per step

VideoTypes = (".avi", ".mp4", ".mov", ".mkv")


def ListVideos(Folder=None, WithDuration=True):
    """Table of the videos in ``Folder`` (default: the calving videos folder):
    file name, size, length, and whether the pipeline already has an output
    folder for it."""
    import pandas as pd

    Folder = Path(Folder or RawVideoDir)
    if not Folder.exists():
        raise FileNotFoundError(f"{Folder} does not exist. Is Drive mounted, and was the folder renamed? "
                                f"Folders in {GroupDir}: {[P.name for P in GroupDir.iterdir()] if GroupDir.exists() else 'none'}")
    Rows = []
    for P in sorted(Folder.iterdir()):
        if P.suffix.lower() not in VideoTypes:
            continue
        Row = {"VideoFile": P.name, "SizeGb": round(P.stat().st_size / 2**30, 2), "HasOutputs": (OutputRoot / P.stem).exists()}
        if WithDuration:
            try:
                from .VideoTools import ProbeVideo
                Row["Duration"] = ProbeVideo(P)["Duration"]
            except Exception:
                Row["Duration"] = "?"
        Rows.append(Row)
    return pd.DataFrame(Rows)


def FindVideo(VideoFile) -> Path:
    """Full path of a video from its file name: first in the calving videos
    folder, then anywhere in the group folder. Raises with the list of
    videos that are there if it cannot be found."""
    VideoFile = Path(VideoFile)
    if VideoFile.is_absolute() and VideoFile.exists():
        return VideoFile
    if (RawVideoDir / VideoFile.name).exists():
        return RawVideoDir / VideoFile.name
    Found = sorted(GroupDir.rglob(VideoFile.name)) if GroupDir.exists() else []
    if Found:
        return Found[0]
    There = sorted(P.name for P in RawVideoDir.iterdir()) if RawVideoDir.exists() else []
    raise FileNotFoundError(f"No video called {VideoFile.name!r} in {GroupDir}. "
                            f"Videos in {RawVideoDir.name!r}: {There}")
