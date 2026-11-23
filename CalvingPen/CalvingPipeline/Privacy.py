"""Black out every person in every video on the group drive, so the copies
students and AI tools work with show no faces or bodies of farm staff.

How (``MaskVideo``):

1. Frames are read with ``Reid.Frames.SampleFrames`` (the AVI chunk reader:
   right times, no fake 24-80 h tails) at ``OutFps`` (default 5; the
   cameras change the picture only ~2 times a second, so 5 loses nothing).
   The masked copy is a video, not single pictures: each of its frames is
   checked, because a frame that is not checked could show a person.
2. **Every** frame that is written is checked by YOLO26s (COCO class 0,
   "person") at size 1024 with a low confidence (0.15): a false alarm only
   blacks out a bit of pen, a miss shows a person.
3. Each person box is grown by ``Pad`` (15 %) on every side and more above
   (the head), and held for ``HoldSeconds`` before and after (0.5 s), so a
   frame where YOLO briefly loses someone is still covered.
4. The box is filled black (``Style="Fill"``; "Pixelate" is weaker: coarse
   blocks can still hint at a person). Nothing of the original is kept under it.
5. The masked frames are encoded to a normal H.264 MP4 in ``MaskedVideos/``
   (same folder layout as the originals; ``Crf`` 28: about the size of the
   original), in ``SegmentMinutes`` pieces that are
   joined at the end, so a stopped run continues where it stopped.

Per video, ``MaskedVideos/Logs/<folder>/<video>/`` gets ``Run.json`` (settings,
frames, frames with people), ``PeopleBoxes.csv`` (the detected boxes, no
pictures) and ``MaskCheck.jpg``: masked frames at each new person sighting
and every 15 min, for a person to check that nobody was missed.

Limits: YOLO can miss a person who is mostly hidden, very small, or in dark
night footage. Masking only protects people if students get the
``MaskedVideos`` folder and not the originals (Drive sharing is set by hand).
"""

import json
import os
import subprocess
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .Paths import GroupDir, MaskedVideoRoot, OutputRoot, ParlorOutputRoot, ReidOutputRoot, VideoTypes

CocoPerson = 0
SkipFolders = {OutputRoot.name, ParlorOutputRoot.name, ReidOutputRoot.name, MaskedVideoRoot.name}
BoxColumns = ["TimeSeconds", "X1", "Y1", "X2", "Y2", "Confidence"]


def _Clean(Name: str) -> str:
    return " ".join(Name.split())          # "CalvingPenVideos " -> "CalvingPenVideos"


def MaskedPaths(VideoPath, MaskedRoot=None) -> tuple:
    """(masked MP4, log folder) of an original video in the group folder, or
    in ``MaskedRoot`` (flat: <MaskedRoot>/<video>.mp4) when it is given."""
    VideoPath = Path(VideoPath)
    if MaskedRoot is not None:
        return Path(MaskedRoot) / f"{VideoPath.stem}.mp4", Path(MaskedRoot) / "Logs" / VideoPath.stem
    try:
        Folder = Path(*[_Clean(P) for P in VideoPath.parent.relative_to(GroupDir).parts])
    except ValueError:                      # not under the group folder
        Folder = Path("Other")
    return MaskedVideoRoot / Folder / f"{VideoPath.stem}.mp4", MaskedVideoRoot / "Logs" / Folder / VideoPath.stem


def ListAllVideos(Root=None) -> pd.DataFrame:
    """Every video under the group folder (also through Drive shortcuts),
    except the pipeline outputs and the masked copies themselves."""
    Root, Rows = Path(Root or GroupDir), []
    for Folder, Folders, Files in os.walk(Root, followlinks=True):
        Folders[:] = sorted(F for F in Folders if F not in SkipFolders and not F.startswith("."))
        for File in sorted(Files):
            if Path(File).suffix.lower() not in VideoTypes or File.startswith("."):
                continue
            Source = Path(Folder) / File
            Out, Log = MaskedPaths(Source)
            try:
                Size = Source.stat().st_size / 2**30
            except OSError:
                Size = None
            Run = json.loads((Log / "Run.json").read_text()) if (Log / "Run.json").exists() else {}
            Rows.append({"Video": File, "Folder": str(Path(Folder).relative_to(Root)), "SizeGb": Size and round(Size, 2),
                         "Status": ("trial" if Run.get("Settings", {}).get("EndSeconds") else "done") if Run.get("Finished") else "empty" if Run.get("Empty")
                         else "started" if Run else "to do", "Source": str(Source), "Masked": str(Out)})
    return pd.DataFrame(Rows, columns=["Video", "Folder", "SizeGb", "Status", "Source", "Masked"])


def SplitParts(Videos: pd.DataFrame, Parts: int) -> list:
    """Spread the videos over ``Parts`` Colab runs of about equal size (largest first)."""
    Bins = [[] for _ in range(Parts)]
    Load = [0.0] * Parts
    for Row in Videos.sort_values("SizeGb", ascending=False, na_position="last").itertuples():
        I = int(np.argmin(Load))
        Bins[I].append(Row.Source)
        Load[I] += Row.SizeGb or 0
    return [sorted(B) for B in Bins]


def GrowBox(Box, Width, Height, Pad=0.15, HeadPad=0.25):
    X1, Y1, X2, Y2 = Box[:4]
    W, H = X2 - X1, Y2 - Y1
    return (max(0, int(X1 - Pad * W)), max(0, int(Y1 - (Pad + HeadPad) * H)),
            min(Width, int(np.ceil(X2 + Pad * W))), min(Height, int(np.ceil(Y2 + Pad * H))))


def ApplyMask(Frame, Boxes, Style="Fill", Pad=0.15, HeadPad=0.25):
    """Black (or coarsely pixelated) padded boxes, on a copy of the frame."""
    Out = Frame.copy()
    H, W = Frame.shape[:2]
    for Box in Boxes:
        X1, Y1, X2, Y2 = GrowBox(Box, W, H, Pad, HeadPad)
        if X2 <= X1 or Y2 <= Y1:
            continue
        if Style == "Pixelate":
            Region = Out[Y1:Y2, X1:X2]
            Small = cv2.resize(Region, (max(1, (X2 - X1) // 24), max(1, (Y2 - Y1) // 24)), interpolation=cv2.INTER_AREA)
            Out[Y1:Y2, X1:X2] = cv2.resize(Small, (X2 - X1, Y2 - Y1), interpolation=cv2.INTER_NEAREST)
        else:
            Out[Y1:Y2, X1:X2] = 0
    return Out


class _Encoder:
    """H.264 MP4 writer fed with raw BGR frames through ffmpeg."""

    def __init__(self, Path_, Width, Height, Fps, OutWidth=None, Crf=23, Preset="veryfast"):
        Scale = ["-vf", f"scale={OutWidth}:-2"] if OutWidth else []
        self.Process = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{Width}x{Height}",
             "-r", str(Fps), "-i", "-", *Scale, "-c:v", "libx264", "-preset", Preset, "-crf", str(Crf),
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(Path_)], stdin=subprocess.PIPE)

    def Write(self, Frame):
        self.Process.stdin.write(np.ascontiguousarray(Frame).tobytes())

    def Close(self):
        self.Process.stdin.close()
        if self.Process.wait():
            raise RuntimeError("ffmpeg failed while writing a masked video")


def BestDevice():
    """CUDA (Colab), else the Mac GPU (MPS), else the CPU."""
    import torch
    if torch.cuda.is_available():
        return 0
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _Detect(Model, Frames, ImageSize, Confidence, Device=None):
    Results = Model.predict(Frames, classes=[CocoPerson], conf=Confidence, imgsz=ImageSize, verbose=False,
                            device=Device if Device is not None else BestDevice())
    return [[(*B, round(float(C), 3)) for B, C in zip(R.boxes.xyxy.cpu().numpy().round(1).tolist(), R.boxes.conf.tolist())]
            if R.boxes is not None else [] for R in Results]


def _CheckSheet(Folder: Path, OutPath: Path, Columns=4, Width=480):
    Files = sorted(Folder.glob("T*.jpg"))
    if not Files:
        return None
    Thumbs = [cv2.imread(str(F)) for F in Files]
    Thumbs = [cv2.resize(T, (Width, int(T.shape[0] * Width / T.shape[1]))) for T in Thumbs if T is not None]
    H = max(T.shape[0] for T in Thumbs)
    Thumbs = [cv2.copyMakeBorder(T, 0, H - T.shape[0], 0, 0, cv2.BORDER_CONSTANT) for T in Thumbs]
    Thumbs += [np.zeros_like(Thumbs[0])] * (-len(Thumbs) % Columns)
    Rows = [np.hstack(Thumbs[I:I + Columns]) for I in range(0, len(Thumbs), Columns)]
    cv2.imwrite(str(OutPath), np.vstack(Rows), [cv2.IMWRITE_JPEG_QUALITY, 80])
    return OutPath


def MaskVideo(VideoPath, Model=None, OutFps=5, ModelName="yolo26s.pt", ImageSize=1024, Confidence=0.15, Pad=0.15,
              HeadPad=0.25, HoldSeconds=0.5, Style="Fill", SegmentMinutes=30, OutWidth=None, NativeFps=None,
              BatchFrames=16, CheckEveryMinutes=15, EndSeconds=None, MaskedRoot=None, Device=None, Crf=28,
              Redo=False) -> dict:
    """Masked copy of one video (see the module doc). Returns its Run.json.
    ``EndSeconds``: stop there (a trial; listed as "trial", not "done").
    ``MaskedRoot``: write there instead of the group drive (e.g. on a laptop)."""
    from .Reid.Frames import LocalCopy, SampleFrames

    VideoPath = Path(VideoPath)
    OutPath, LogDir = MaskedPaths(VideoPath, MaskedRoot)
    Device = Device if Device is not None else BestDevice()
    Parts, CheckDir = OutPath.parent / f".{OutPath.stem}.parts", LogDir / "Check"
    Settings = {"Video": VideoPath.name, "OutFps": OutFps, "ModelName": ModelName, "ImageSize": ImageSize,
                "Confidence": Confidence, "Pad": Pad, "HeadPad": HeadPad, "HoldSeconds": HoldSeconds, "Style": Style,
                "SegmentMinutes": SegmentMinutes, "OutWidth": OutWidth, "Crf": Crf, "NativeFps": NativeFps, "EndSeconds": EndSeconds}
    RunPath = LogDir / "Run.json"
    Run = json.loads(RunPath.read_text()) if RunPath.exists() else {}
    Same = Run.get("Settings") == Settings and not Redo
    if Same and (Run.get("Finished") or Run.get("Empty")):
        print(f"{VideoPath.name}: already {'done' if Run.get('Finished') else 'checked (empty)'}; Redo=True to redo")
        return Run
    if not Same:
        for Old in list(Parts.glob("*.mp4")) + list(CheckDir.glob("*.jpg")) + [LogDir / "PeopleBoxes.csv"]:
            Old.unlink(missing_ok=True)
        Run = {"Settings": Settings, "Finished": False, "DoneSegments": 0, "Frames": 0, "FramesWithPeople": 0}
    for Folder in (Parts, LogDir, CheckDir):
        Folder.mkdir(parents=True, exist_ok=True)
    SegSeconds = SegmentMinutes * 60
    From = Run["DoneSegments"] * SegSeconds
    BoxFile = LogDir / "PeopleBoxes.csv"
    if BoxFile.exists():                    # drop boxes of a segment that was not finished
        Old = pd.read_csv(BoxFile)
        Old[Old.TimeSeconds < From].to_csv(BoxFile, index=False)
    else:
        pd.DataFrame(columns=BoxColumns).to_csv(BoxFile, index=False)

    if Model is None:
        from ultralytics import YOLO
        Model = YOLO(ModelName)
    Hold = max(0, int(round(HoldSeconds * OutFps)))
    Pending, Past = deque(), deque(maxlen=Hold)
    State = {"Segment": None, "Encoder": None, "LastSighting": -1e9, "NextCheck": From}
    Info, Started = {}, time.time()
    print(f"{VideoPath.name}: {OutFps} fps from {From / 60:.0f} min on {Device}, people blacked out -> {OutPath}")

    def Close(Finished):
        if State["Encoder"] is not None:
            State["Encoder"].Close()
            State["Encoder"] = None
            if Finished:
                Run["DoneSegments"] = State["Segment"] + 1
                RunPath.write_text(json.dumps(Run, indent=1))

    def Emit():
        T, Frame, Boxes = Pending.popleft()
        Cover = Boxes + [B for P in Past for B in P] + [B for _, _, P in list(Pending)[:Hold] for B in P]
        Out = ApplyMask(Frame, Cover, Style, Pad, HeadPad)
        Past.append(Boxes)
        Segment = int(T // SegSeconds)
        if Segment != State["Segment"]:
            Close(Finished=True)
            State["Segment"] = Segment
            State["Encoder"] = _Encoder(Parts / f"Part{Segment:03d}.mp4", Out.shape[1], Out.shape[0], OutFps, OutWidth, Crf)
        State["Encoder"].Write(Out)
        Run["Frames"] += 1
        if Cover:
            Run["FramesWithPeople"] += 1
        New = bool(Boxes) and T - State["LastSighting"] > 60
        if Boxes:
            State["LastSighting"] = T
        if New or T >= State["NextCheck"]:          # masked frames for the check sheet
            if T >= State["NextCheck"]:
                State["NextCheck"] = T + CheckEveryMinutes * 60
            Small = cv2.resize(Out, (480, int(Out.shape[0] * 480 / Out.shape[1])))
            cv2.putText(Small, f"{int(T // 3600)}:{int(T % 3600 // 60):02d}:{int(T % 60):02d}" + ("  person" if New else ""),
                        (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.imwrite(str(CheckDir / f"T{int(T):06d}.jpg"), Small)

    def Flush(Batch):
        Unique = {}
        for Tick, _, Frame in Batch:
            Unique.setdefault(Tick, Frame)       # a held picture is checked once
        Found = dict(zip(Unique, _Detect(Model, list(Unique.values()), ImageSize, Confidence, Device)))
        Rows = []
        for Tick, T, Frame in Batch:
            Pending.append((T, Frame, Found[Tick]))
            Rows += [[T, *B] for B in Found[Tick]]
            while len(Pending) > Hold:
                Emit()
        if Rows:
            pd.DataFrame(Rows, columns=BoxColumns).to_csv(BoxFile, mode="a", header=False, index=False)

    try:
        Batch = []
        for Tick, T, Frame in SampleFrames(LocalCopy(VideoPath), OutFps, From, EndSeconds, NativeFps, Info):
            Batch.append((Tick, T, Frame))
            if len(Batch) == BatchFrames:
                Flush(Batch)
                Batch = []
        if Batch:
            Flush(Batch)
        while Pending:
            Emit()
        Close(Finished=True)
    finally:
        Close(Finished=False)

    if (Info.get("RealPictures") or 0) <= 1 and From == 0:
        Run.update({"Empty": True, "RealPictures": Info.get("RealPictures")})
        RunPath.write_text(json.dumps(Run, indent=1))
        for P in Parts.glob("*.mp4"):
            P.unlink()
        Parts.rmdir()
        print(f"{VideoPath.name}: no real video in it (an empty recording); nothing written")
        return Run
    Files = sorted(Parts.glob("Part*.mp4"))
    List = Parts / "List.txt"
    List.write_text("".join(f"file '{F.name}'\n" for F in Files))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(List), "-c", "copy",
                    "-movflags", "+faststart", str(OutPath)], check=True)
    for F in Files + [List]:
        F.unlink()
    Parts.rmdir()
    _CheckSheet(CheckDir, LogDir / "MaskCheck.jpg")
    Run.update({"Finished": True, "LastRealSeconds": Info.get("LastRealSeconds"), "RealPictures": Info.get("RealPictures"),
                "Minutes": round((time.time() - Started) / 60, 1),
                "PeopleShare": round(Run["FramesWithPeople"] / max(Run["Frames"], 1), 4)})
    RunPath.write_text(json.dumps(Run, indent=1))
    print({K: Run.get(K) for K in ("Frames", "FramesWithPeople", "PeopleShare", "LastRealSeconds", "Minutes")})
    return Run


def MaskAll(Videos=None, Part=None, Parts=4, Redo=False, **Settings) -> pd.DataFrame:
    """Masks every video of ``ListAllVideos()`` (or the ``Videos`` given), or
    only ``Part`` (1..``Parts``) of them, so several people can run it at once.
    One video failing does not stop the others. Returns a table of results."""
    from ultralytics import YOLO

    Table = ListAllVideos() if Videos is None else pd.DataFrame({"Source": [str(V) for V in Videos], "SizeGb": None})
    Sources = SplitParts(Table, Parts)[Part - 1] if Part else sorted(Table.Source)
    print(f"{len(Sources)} videos" + (f" (part {Part} of {Parts})" if Part else ""))
    Model, Results = YOLO(Settings.get("ModelName", "yolo26s.pt")), []
    for I, Source in enumerate(Sources, 1):
        print(f"\n[{I}/{len(Sources)}] {Source}")
        try:
            Run = MaskVideo(Source, Model=Model, Redo=Redo, **Settings)
            Results.append({"Video": Path(Source).name, "Status": "empty" if Run.get("Empty") else
                            "trial" if Settings.get("EndSeconds") else "done",
                            "PeopleShare": Run.get("PeopleShare"),
                            "Masked": str(MaskedPaths(Source, Settings.get("MaskedRoot"))[0])})
        except Exception as Error:          # e.g. a broken file: note it, go on with the next video
            print(f"  FAILED: {type(Error).__name__}: {Error}")
            Results.append({"Video": Path(Source).name, "Status": f"failed: {Error}", "PeopleShare": None,
                            "Masked": None})
        finally:
            from .Reid.Frames import LocalCopy
            Local = LocalCopy.__defaults__[0]
            Copy = Path(Local) / Path(Source).name
            if Copy.exists() and Copy != Path(Source):
                Copy.unlink()              # free the Colab disk for the next video
    return pd.DataFrame(Results)


def Main(Arguments=None):
    """Terminal use, e.g. on a laptop with videos already downloaded:

        python -m CalvingPipeline.Privacy Video1.avi Video2.mp4 --minutes 30 --out ~/MaskedVideos

    Writes <out>/<video>.mp4 and <out>/Logs/<video>/MaskCheck.jpg."""
    import argparse

    Parser = argparse.ArgumentParser(prog="python -m CalvingPipeline.Privacy",
                                     description="Black out every person in videos (see Privacy.py).")
    Parser.add_argument("Videos", nargs="+", help="video files (.avi / .mp4 / ...)")
    Parser.add_argument("--out", default=str(Path.home() / "MaskedVideos"), help="output folder (default ~/MaskedVideos)")
    Parser.add_argument("--minutes", type=float, default=None, help="only the first N minutes (a trial)")
    Parser.add_argument("--fps", type=float, default=5, help="frames per second of the masked copy (default 5)")
    Parser.add_argument("--model", default="yolo26s.pt", help="person detector (default yolo26s.pt)")
    Parser.add_argument("--size", type=int, default=1024, help="YOLO picture size (default 1024)")
    Parser.add_argument("--device", default=None, help="cpu, mps or 0 (default: the best there is)")
    Parser.add_argument("--redo", action="store_true", help="redo videos that are done")
    A = Parser.parse_args(Arguments)
    Out = Path(A.out).expanduser().resolve()
    for V in A.Videos:
        if not Path(V).expanduser().is_file():
            raise SystemExit(f"No video at {V}: check the name (drag the file from Finder into Terminal to paste its path)")
        if Out in Path(V).resolve().parents or Path(V).resolve().parent == Out:
            raise SystemExit(f"--out must not contain the input videos ({V})")
    Results = MaskAll(Videos=[Path(V).expanduser() for V in A.Videos], Redo=A.redo, MaskedRoot=Out, OutFps=A.fps,
                      ModelName=A.model, ImageSize=A.size, Device=A.device,
                      EndSeconds=A.minutes * 60 if A.minutes else None)
    print(Results.to_string(index=False))
    print(f"Check sheets: {Out / 'Logs'}/<video>/MaskCheck.jpg")


if __name__ == "__main__":
    Main()
