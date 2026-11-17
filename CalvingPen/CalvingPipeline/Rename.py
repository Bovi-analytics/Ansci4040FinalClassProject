"""Give the calving videos (and their output folders) one naming convention, safely.

The rule (agreed by the group, from RenameCalvingVideos.ipynb):

    CalvingPen<pen><Mon><day>Cow<n>[ROI]

- ``<pen>``: the pen number; ``<Mon><day>``: month and day, no year, day
  without a leading zero (``Nov11``, ``Sep6``); no start time,
- ``Cow<n>``: the recording number in brackets in the old name (``Pen 3 (2)``
  -> ``Cow2``), 1 if there is none,
- ``ROI``: added when the old name contains ROI (a cropped clip).

    2026-11-11 00-00-00 Calving Pen 3 (2).avi                   -> CalvingPen3Nov11Cow2.avi
    2026-11-11 00-00-00 Calving Pen 3.avi                       -> CalvingPen3Nov11Cow1.avi
    2026-10-29 09-48-06 Calving Pen 2 (3).avi                   -> CalvingPen2Oct29Cow3.avi
    2026-09-26-CalvingPen3(1)_ROI_01-42-00_to_01-57-00_2FPS.avi -> CalvingPen3Sep26Cow1ROI.avi

Names with ``Clip`` (``2026-09-26-Clip1Pen2(4:37hrs).avi``) or without a date
or pen are marked ``Check`` and left alone (rename by hand). Two files that would
get the same name (e.g. two ROI clips of one video, a "Copy of ...") are marked
``Conflict`` and left alone.

**Output folders** in ``CalvingPenPipelineOutputs`` get the same rule
(``2026-09-26 00-00-00 Calving Pen 2`` -> ``CalvingPen2Sep26Cow1``; a window
folder keeps its ``Win1`` ending), and the paths and names stored inside them
(settings files, ``EmbeddingIndex.csv``) are updated, so finished steps stay
finished.

Safe by design:

- ``PlanRenames`` / ``PlanFolderRenames`` only **preview** (old name, new name, status),
- ``ApplyRenames`` renames only rows with status ``Rename`` (two passes via
  temporary names) and appends every rename to ``RenameLog.csv``,
- ``UndoRenames`` puts the names from the log back (folders included),
- ``FindVideo`` (Paths.py) finds a video by its old name through the log, and
  by its new name before it was renamed.
"""

import re
import time
import uuid
from pathlib import Path

import pandas as pd

from .Paths import GroupDir, OutputRoot, VideoTypes

RenameLog = OutputRoot / "RenameLog.csv"
LogColumns = ["Time", "Folder", "OldName", "NewName"]
Months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
NewStyle = re.compile(r"^CalvingPen(\d+)(" + "|".join(Months) + r")(\d{1,2})Cow(\d+)(ROI)?(Win\d+)?$")
PartStyle = re.compile(r"^CalvingPen\(?(\d+)\)?(" + "|".join(Months) + r")(\d{1,2})(?:Time\d{6})?Part\(?(\d+)\)?(.*)$")


def MonthDay(Year, Month, Day) -> str:
    """("2026", "11", "01") -> "Nov1" (no year, no leading zero)."""
    return f"{Months[int(Month) - 1]}{int(Day)}"


def CalvingName(FileName: str, Brackets=False) -> tuple:
    """``(NewName, Status)`` for one file or folder name; Status "Ok" or "Check".
    (``Brackets`` is ignored; kept so older notebooks still run.)"""
    P = Path(FileName)
    Ext = P.suffix.lower() if P.suffix.lower() in VideoTypes else ""
    Name = P.stem if Ext else P.name
    Win = re.search(r"(Win\d+)$", Name)
    WinText = Win.group(1) if Win else ""
    if NewStyle.match(Name):                                   # already in the convention
        return Name + Ext, "Ok"
    Old = PartStyle.match(Name)                                # the short-lived CalvingPen3Nov11Part1 style
    if Old:
        Pen, Mon, Day, Part, Rest = Old.groups()
        Roi = "ROI" if re.search(r"roi", Rest, re.I) else ""
        return f"CalvingPen{Pen}{Mon}{int(Day)}Cow{Part}{Roi}{WinText}" + Ext, "Ok"
    if re.search(r"Clip\d+", Name, re.I):                     # clip files need a person to decide
        return Name + Ext, "Check"
    PenMatch = re.search(r"Calving[\s_-]*Pen[\s_-]*(\d+)|Pen[\s_-]*(\d+)", Name, re.I)
    DateMatch = re.search(r"(\d{4})[-_](\d{2})[-_](\d{2})", Name)
    if not (PenMatch and DateMatch) or not (1 <= int(DateMatch.group(2)) <= 12):
        return Name + Ext, "Check"
    Pen = next(G for G in PenMatch.groups() if G is not None)
    CowMatch = re.search(r"\((\d+)\)", Name)                 # recording number in brackets = cow number
    Cow = int(CowMatch.group(1)) if CowMatch else 1
    Roi = "ROI" if re.search(r"ROI", Name, re.I) else ""
    return f"CalvingPen{Pen}{MonthDay(*DateMatch.groups())}Cow{Cow}{Roi}{WinText}" + Ext, "Ok"


def PlanRenames(Folders, Brackets=False, Recursive=True) -> pd.DataFrame:
    """Preview: one row per video file in ``Folders`` (names relative to the group
    folder, e.g. "CalvingPenVideos "), with the new name and what will happen:
    Rename / Same (already right) / Check (name not understood, left alone) /
    Unreadable (broken shortcut or uploading, left alone) / Conflict (new name
    taken or used twice, left alone)."""
    Rows = []
    for Folder in Folders:
        Root = Path(Folder) if Path(Folder).is_absolute() else GroupDir / Folder
        if not Root.exists():
            raise FileNotFoundError(f"{Root} does not exist; folders in {GroupDir}: "
                                    f"{sorted(P.name for P in GroupDir.iterdir()) if GroupDir.exists() else 'none'}")
        Files = Root.rglob("*") if Recursive else Root.iterdir()
        for P in sorted(Files):
            if P.suffix.lower() not in VideoTypes or P.name.startswith(".rename_temp_"):
                continue
            New, Status = CalvingName(P.name)
            try:
                P.stat()
            except OSError:
                Status = "Unreadable"
            Rows.append({"Folder": str(P.parent.relative_to(GroupDir)) if P.is_relative_to(GroupDir) else str(P.parent),
                         "OldName": P.name, "NewName": New,
                         "Status": Status if Status != "Ok" else ("Same" if New == P.name else "Rename")})
    return _MarkConflicts(pd.DataFrame(Rows, columns=["Folder", "OldName", "NewName", "Status"]))


def PlanFolderRenames(Root=None) -> pd.DataFrame:
    """Preview for the video folders in ``CalvingPenPipelineOutputs`` (Pool, PoolTest,
    Old and files are left out): same rule and statuses as ``PlanRenames``."""
    Root = Path(Root or OutputRoot)
    Rows = []
    for P in sorted(Root.iterdir()):
        if not P.is_dir() or P.name in ("Pool", "PoolTest", "Old") or P.name.startswith("."):
            continue
        New, Status = CalvingName(P.name)
        Rows.append({"Folder": str(Root.relative_to(GroupDir)) if Root.is_relative_to(GroupDir) else str(Root),
                     "OldName": P.name, "NewName": New,
                     "Status": Status if Status != "Ok" else ("Same" if New == P.name else "Rename")})
    return _MarkConflicts(pd.DataFrame(Rows, columns=["Folder", "OldName", "NewName", "Status"]))


def _MarkConflicts(Plan):
    if len(Plan):   # two files would get the same name, or the name is already taken by another file
        Key = Plan.Folder + "/" + Plan.NewName
        Twice = Key.duplicated(keep=False) & Plan.Status.eq("Rename")
        for _, Group in Plan[Twice].groupby(Key[Twice]):          # an original and its "Copy of ...": rename the original
            Originals = Group.index[~Group.OldName.str.match(r"(?i)\s*copy of")]
            if len(Originals) == 1:
                Twice[Originals[0]] = False
        Taken = Plan.apply(lambda R: R.Status == "Rename" and (GroupDir / R.Folder / R.NewName).exists(), axis=1)
        Plan.loc[Twice | Taken, "Status"] = "Conflict"
    return Plan


def _FixFolder(Dir, OldName, NewName):
    """After a video folder was renamed: update the paths stored in its settings
    files and the VideoStem column of its fingerprint index, so finished steps
    are still recognised as finished."""
    Dir = Path(Dir)
    for Json in Dir.rglob("*.json"):
        Text = Json.read_text()
        Fixed = Text.replace(f"/{OldName}/", f"/{NewName}/").replace(f"/{OldName}\"", f"/{NewName}\"")
        if Fixed != Text:
            Json.write_text(Fixed)
    for Index in Dir.glob("Step05*/EmbeddingIndex.csv"):
        Table = pd.read_csv(Index)
        if "VideoStem" in Table:
            Table["VideoStem"] = Table.VideoStem.replace(OldName, NewName)
            Table.to_csv(Index, index=False)


def ApplyRenames(Plan: pd.DataFrame, LogPath=RenameLog) -> pd.DataFrame:
    """Rename every row with Status "Rename" and log it. Returns the rows done."""
    Todo = Plan[Plan.Status == "Rename"]
    Temporary = []
    for R in Todo.itertuples():                       # pass 1: to unique temporary names
        Old = GroupDir / R.Folder / R.OldName
        Temp = Old.with_name(f".rename_temp_{uuid.uuid4().hex}{Old.suffix}")
        Old.rename(Temp)
        Temporary.append((Temp, Old.with_name(R.NewName), R))
    Done = []
    for Temp, New, R in Temporary:                     # pass 2: to the new names
        Temp.rename(New)
        if New.is_dir():
            _FixFolder(New, R.OldName, R.NewName)
        Done.append({"Time": time.strftime("%Y-%m-%d %H:%M:%S"), "Folder": R.Folder,
                     "OldName": R.OldName, "NewName": R.NewName})
    Done = pd.DataFrame(Done, columns=LogColumns)
    if len(Done):
        Path(LogPath).parent.mkdir(parents=True, exist_ok=True)
        Done.to_csv(LogPath, mode="a", index=False, header=not Path(LogPath).exists())
    print(f"Renamed {len(Done)}; logged in {LogPath}")
    return Done


def UndoRenames(LogPath=RenameLog, Since=None) -> int:
    """Give files their old names back (newest first), for renames logged at or
    after ``Since`` ("YYYY-MM-DD HH:MM:SS"; None = all). Removes them from the log."""
    Log = pd.read_csv(LogPath)
    Undo = Log if Since is None else Log[Log.Time >= Since]
    Count = 0
    for R in Undo.iloc[::-1].itertuples():
        New, Old = GroupDir / R.Folder / R.NewName, GroupDir / R.Folder / R.OldName
        if New.exists() and not Old.exists():
            New.rename(Old)
            if Old.is_dir():
                _FixFolder(Old, R.NewName, R.OldName)
            Count += 1
    Log.drop(Undo.index).to_csv(LogPath, index=False)
    print(f"Put {Count} names back")
    return Count


def RenamedTo(OldName, LogPath=None):
    """Current path of a video that was renamed from ``OldName`` (via the log), or None."""
    from . import Paths
    LogPath = LogPath or Paths.OutputRoot / "RenameLog.csv"
    GroupDir = Paths.GroupDir
    if not Path(LogPath).exists():
        return None
    Log = pd.read_csv(LogPath)
    Name = Path(OldName).name
    for _ in range(10):                                # follow renames of renames
        Hit = Log[Log.OldName == Name]
        if Hit.empty:
            break
        Folder, Name = Hit.Folder.iloc[-1], Hit.NewName.iloc[-1]
        if (GroupDir / Folder / Name).exists():
            return GroupDir / Folder / Name
    return None
