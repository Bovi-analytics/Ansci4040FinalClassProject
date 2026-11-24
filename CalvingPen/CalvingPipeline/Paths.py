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
ParlorVideoDir = GroupDir / "EntryParlorVideos"
ParlorOutputRoot = GroupDir / "ParlorPipelineOutputs"   # same layout, for the milking-parlor videos
ReidOutputRoot = GroupDir / "ReidPipelineOutputs"       # re-ID design v2: one folder per validation, then per video
MaskedVideoRoot = GroupDir / "MaskedVideos"             # people blacked out (Privacy.py): the copies students use
BoviSyncDir = GroupDir / "BoviSyncRecords"              # farm records: calvings (Freshen events), parlor reports

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
        try:
            Size = round(P.stat().st_size / 2**30, 2)
        except OSError:   # listed but not readable: a broken Drive shortcut, or still uploading / being moved
            Rows.append({"VideoFile": P.name, "SizeGb": None, "HasOutputs": (OutputRoot / P.stem).exists(),
                         "Duration": "not readable (broken shortcut or still uploading?)"})
            continue
        Row = {"VideoFile": P.name, "SizeGb": Size, "HasOutputs": (OutputRoot / P.stem).exists()}
        if WithDuration:
            try:
                from .VideoTools import ProbeVideo
                Row["Duration"] = ProbeVideo(P)["Duration"]
            except Exception:
                Row["Duration"] = "?"
        Rows.append(Row)
    return pd.DataFrame(Rows)


def _Loose(Name: str) -> str:
    """Name compared without case, extra spaces or the extension."""
    return " ".join(Path(Name).stem.lower().split())


def FindVideo(VideoFile) -> Path:
    """Full path of a video from its file name (or a full path): first in the
    calving videos folder, then anywhere in the group folder, also through
    Drive shortcuts; case, double spaces and the extension do not matter.
    Old names work after renaming (RenameLog.csv) and new PascalCase names
    (CalvingPen3Nov11Cow1.avi) work before renaming.
    Raises with the videos that are there if it cannot be found."""
    import os

    VideoFile = Path(str(VideoFile).strip())
    if VideoFile.exists():                                   # a full path that works
        return VideoFile
    if (RawVideoDir / VideoFile.name).exists():
        return RawVideoDir / VideoFile.name
    from .Rename import RenamedTo                            # renamed to the PascalCase convention?
    Renamed = RenamedTo(VideoFile.name)
    if Renamed is not None:
        print(f"{VideoFile.name} was renamed to {Renamed.name}")
        return Renamed
    from .Rename import CalvingName
    Wanted, Similar = _Loose(VideoFile.name), []
    for Root, Folders, Files in os.walk(GroupDir, followlinks=True):   # followlinks: Drive shortcuts
        Folders[:] = [F for F in Folders if F != OutputRoot.name]        # skip the pipeline outputs (thousands of crops)
        Folders.sort(key=lambda F: (F != RawVideoDir.name, not F.startswith("Calving"), F))
        for File in Files:
            if Path(File).suffix.lower() not in VideoTypes:
                continue
            # the name itself, or its new name (CalvingPen3Nov11Cow1) before it was renamed
            if _Loose(File) == Wanted or _Loose(CalvingName(File)[0]) == Wanted:
                return Path(Root) / File
            if Wanted[:10] and _Loose(File).startswith(Wanted[:10]):
                Similar.append(str(Path(Root) / File))
    There = sorted(P.name for P in RawVideoDir.iterdir()) if RawVideoDir.exists() else []
    raise FileNotFoundError(
        f"No video called {VideoFile.name!r} anywhere in {GroupDir}.\n"
        f"  Similar names: {Similar[:10] or 'none'}\n"
        f"  Videos in {RawVideoDir.name!r}: {There}\n"
        "  Fix: copy the exact name from ListVideos(), or paste the video's full path as VideoFile.")


# ---- repeated videos (the same file uploaded twice, a "Copy of ...", an old and a new name) ----

def VideoFingerprint(VideoPath, SampleBytes=2**20) -> str:
    """Size + md5 of 1 MB at the start, middle and end: equal for two copies of
    one file, different for different recordings. Reads 3 MB, not the 4 GB."""
    import hashlib

    VideoPath = Path(VideoPath)
    Size = VideoPath.stat().st_size
    Hash = hashlib.md5(str(Size).encode())
    with open(VideoPath, "rb") as F:
        for Start in (0, max(0, Size // 2 - SampleBytes // 2), max(0, Size - SampleBytes)):
            F.seek(Start)
            Hash.update(F.read(SampleBytes))
    return f"{Size}-{Hash.hexdigest()[:16]}"


def _KeepScore(P: Path):
    """Which copy to keep: a PascalCase name (CalvingPen3Nov11Cow2), in the
    videos folder, not "Copy of", then the shortest path."""
    Name = P.name
    return (not Name.startswith("CalvingPen"), "copy" in Name.lower(), P.parent.name != RawVideoDir.name,
            len(str(P)), str(P))


def FindRepeats(Paths_=None, Root=None) -> "pd.DataFrame":
    """Videos that are byte-for-byte the same as another one (``Paths_``, or every
    video under the group folder except the pipeline outputs and masked copies).
    One row per file: ``Fingerprint``, ``Keep`` (True for the copy to use) and
    ``RepeatOf`` (the kept copy's path). Empty (0-byte) files are listed with
    ``RepeatOf = "empty file"``. Nothing is deleted: delete the ``Keep = False``
    rows by hand on Drive."""
    import os

    import pandas as pd

    if Paths_ is None:
        Skip = {OutputRoot.name, ParlorOutputRoot.name, ReidOutputRoot.name, MaskedVideoRoot.name}
        Paths_ = []
        for Folder, Folders, Files in os.walk(Root or GroupDir, followlinks=True):
            Folders[:] = [F for F in Folders if F not in Skip and not F.startswith(".")]
            Paths_ += [Path(Folder) / F for F in Files if Path(F).suffix.lower() in VideoTypes]
    Rows = []
    for P in map(Path, Paths_):
        try:
            Size = P.stat().st_size
        except OSError:
            continue
        Rows.append({"Path": P, "Video": P.name, "Size": Size})
    Table = pd.DataFrame(Rows, columns=["Path", "Video", "Size"])
    if Table.empty:
        return Table.assign(Fingerprint=[], Keep=[], RepeatOf=[])
    Same = Table.Size.duplicated(keep=False) & (Table.Size > 0)      # only files of equal size can be copies
    Table["Fingerprint"] = [VideoFingerprint(P) if S else f"{Z}" for P, Z, S in zip(Table.Path, Table.Size, Same)]
    Table["Keep"], Table["RepeatOf"] = True, ""
    for _, Group in Table[Same].groupby("Fingerprint"):
        if len(Group) < 2:
            continue
        Best = min(Group.Path, key=_KeepScore)
        for I, P in zip(Group.index, Group.Path):
            if P != Best:
                Table.loc[I, ["Keep", "RepeatOf"]] = [False, str(Best)]
    Empty = Table.Size == 0
    Table.loc[Empty, ["Keep", "RepeatOf"]] = [False, "empty file"]
    Table["Path"] = Table.Path.astype(str)
    return Table[~Table.Keep | Table.Fingerprint.isin(Table[~Table.Keep].Fingerprint)].sort_values(
        ["Fingerprint", "Keep"], ascending=[True, False]).reset_index(drop=True)
