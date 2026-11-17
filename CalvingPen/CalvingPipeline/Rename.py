"""Give the calving videos one PascalCase naming convention, safely.

The raw files have many name styles, e.g.

    2026-11-11 00-00-00 Calving Pen 3 (2).avi
    2026-09-26-CalvingPen3(1).avi
    2026-10-29_09-48-06_Calving_Pen_2_(3)ROI.mp4
    2026-09-26-CalvingPen3(1)_ROI_01-42-00_to_01-57-00_2FPS.avi

``CalvingName`` turns each into one pattern, keeping the date, the start
time (only when it is not midnight), the pen and the part number:

    CalvingPen3_20261111_Part2.avi
    CalvingPen3_20260926_Part1.avi
    CalvingPen2_20261029_094806_Part3_Roi.mp4
    CalvingPen3_20260926_Part1_Roi_014200_To_015700_2Fps.avi

(``Brackets=True``: ``CalvingPen(3)_20261111_Part(2).avi``.) A file without a
date or pen in its name only gets PascalCase and is marked ``Check``.

Safe by design:

- ``PlanRenames`` only **previews** (old name, new name, status); nothing changes,
- ``ApplyRenames`` renames only rows with status ``Rename`` (in two passes via
  temporary names, so two files can swap names), and appends every rename to
  ``RenameLog.csv`` (time, folder, old, new),
- ``UndoRenames`` puts the names from the log back,
- ``FindVideo`` (Paths.py) also finds a video by its **old** name through the
  log, so notebooks with old names keep working.

Only video files are touched (``.avi .mp4 .mov .mkv``); notebooks, spreadsheets
and pipeline outputs keep their names, so nothing that refers to them breaks.
"""

import re
import time
import uuid
from pathlib import Path

import pandas as pd

from .Paths import GroupDir, OutputRoot, VideoTypes

RenameLog = OutputRoot / "RenameLog.csv"
NewStyle = re.compile(r"^CalvingPen(\d+|\(\d+\))_\d{8}(_\d{6})?_Part(\d+|\(\d+\))(_[A-Za-z0-9]+)*$")
LogColumns = ["Time", "Folder", "OldName", "NewName"]


def _Pascal(Text: str) -> list:
    """Words / numbers of a leftover name part, PascalCase, time-like groups joined
    (01-42-00 -> 014200)."""
    Text = re.sub(r"(\d{2})[-:_](\d{2})[-:_](\d{2})", r"\1\2\3", Text)
    Text = re.sub(r"(\d+):(\d{2})\s*hrs?", r"Length\1h\2m", Text, flags=re.I)    # "4:37hrs" -> Length4h37m
    Words = []
    for W in re.findall(r"[A-Za-z0-9]+", Text):
        Digits = re.match(r"\d*", W).group()
        Words.append(Digits + W[len(Digits):].capitalize() if Digits else W[0].upper() + W[1:].lower())
    return Words


def CalvingName(FileName: str, Brackets=False) -> tuple:
    """``(NewName, Status)`` for one file name; Status is "Ok" or "Check"
    (no date or pen found: only made PascalCase)."""
    P = Path(FileName)
    Ext, Stem = P.suffix.lower(), P.stem
    if NewStyle.match(Stem):                       # already in the convention: keep it
        return Stem + Ext, "Ok"
    Stem = re.sub(r"^\s*Copy of\s+", "", Stem, flags=re.I)
    IsCopy = Stem != P.stem
    Date = re.search(r"(20\d{2})[-_ ]?(\d{2})[-_ ]?(\d{2})", Stem)
    Pen = re.search(r"Pen[\s_-]*\(?(\d+)\)?", Stem, flags=re.I)
    if not (Date and Pen):
        Words = _Pascal(Stem) + (["Copy"] if IsCopy else [])
        return "".join(Words) + Ext, "Check"
    Rest = Stem
    Time = re.match(r"[\s_-]*(\d{2})[-_:](\d{2})[-_:](\d{2})", Stem[Date.end():])
    TimeText = "" if not Time or Time.groups() == ("00", "00", "00") else "".join(Time.groups())
    # Part: a number in brackets right after the pen, e.g. "Pen 3 (2)" / "Pen2_(3)"
    PartMatch = re.match(r"[\s_-]*\(\s*(\d+)\s*\)", Stem[Pen.end():])
    ClipMatch = re.search(r"Clip\s*(\d+)", Stem, flags=re.I)        # "Clip1Pen2" = part 1
    Part = PartMatch.group(1) if PartMatch else (ClipMatch.group(1) if ClipMatch else "1")
    # Everything else (ROI, time window, fps, ...) is kept, in PascalCase.
    Cut = [Date.span(), Pen.span()]
    if Time:
        Cut.append((Date.end(), Date.end() + Time.end()))
    if PartMatch:
        Cut.append((Pen.end(), Pen.end() + PartMatch.end()))
    for Start, End in sorted(Cut, reverse=True):
        Rest = Rest[:Start] + " " + Rest[End:]
    Rest = re.sub(r"Calving|Clip\s*\d*", " ", Rest, flags=re.I)
    PenText, PartText = (f"({Pen.group(1)})", f"({Part})") if Brackets else (Pen.group(1), Part)
    Fields = [f"CalvingPen{PenText}", "".join(Date.groups())] + ([TimeText] if TimeText else []) + [f"Part{PartText}"]
    Fields += _Pascal(Rest) + (["Copy"] if IsCopy else [])
    return "_".join(Fields) + Ext, "Ok"


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
            New, Status = CalvingName(P.name, Brackets)
            try:
                P.stat()
            except OSError:
                Status = "Unreadable"
            Rows.append({"Folder": str(P.parent.relative_to(GroupDir)) if P.is_relative_to(GroupDir) else str(P.parent),
                         "OldName": P.name, "NewName": New,
                         "Status": "Same" if New == P.name else ("Rename" if Status == "Ok" else Status)})
    Plan = pd.DataFrame(Rows, columns=["Folder", "OldName", "NewName", "Status"])
    if len(Plan):   # two files would get the same name, or the name is already taken by another file
        Key = Plan.Folder + "/" + Plan.NewName
        Twice = Key.duplicated(keep=False) & Plan.Status.eq("Rename")
        Taken = Plan.apply(lambda R: R.Status == "Rename" and (GroupDir / R.Folder / R.NewName).exists(), axis=1)
        Plan.loc[Twice | Taken, "Status"] = "Conflict"
    return Plan


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
        Done.append({"Time": time.strftime("%Y-%m-%d %H:%M:%S"), "Folder": R.Folder,
                     "OldName": R.OldName, "NewName": R.NewName})
    Done = pd.DataFrame(Done, columns=LogColumns)
    if len(Done):
        Path(LogPath).parent.mkdir(parents=True, exist_ok=True)
        Done.to_csv(LogPath, mode="a", index=False, header=not Path(LogPath).exists())
    print(f"Renamed {len(Done)} files; logged in {LogPath}")
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
