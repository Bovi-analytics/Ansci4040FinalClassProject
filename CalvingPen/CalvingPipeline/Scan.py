"""Step 01 (automatic): find when cows are in view in a whole raw recording,
and pick a few windows to run the pipeline on.

Tracking (step 03) costs ~1.25x the clip length on a T4, so a 4.5 h recording
would take ~6 h. For recognising a cow, a few 10-minute windows from different
times of the recording (different poses and light) are enough. Here:

1. ``ScanVideo``: YOLO26n looks at one frame every ``EverySeconds`` (30 s) of
   the **whole raw video**. ffmpeg decodes only key frames (``-skip_frame
   nokey``), so the 4 GB file is read once without decoding every frame:
   a few minutes per recording.
2. ``PickWindows``: per minute, the share of looked-at frames with a confident
   cow (``MinConfidence``); windows of ``WindowMinutes`` where that share is
   at least ``MinCowShare``; the recording is split into ``Windows`` equal
   parts and the best window of each part is taken (so the windows are spread
   out). ``MaxCows`` says if a window shows more than one cow.
3. ``AutoWindows`` does both, saves the results, and reuses them on a rerun.

Outputs in ``<OutputRoot>/<VideoStem>/Scan/``:

- ``Scan.csv``: one row per looked-at frame (TimeSeconds, Cows, BestConfidence)
- ``Windows.csv``: the chosen windows (Window, ClipStart, ClipEnd, CowShare, MaxCows)
- ``CowTimeline.png``: cows in view over the recording, chosen windows shaded
- ``Scan.json``: the settings (a rerun with the same settings reuses the scan)

Each window then becomes its own pipeline run, output folder ``<VideoStem>Win1``,
``Win2``, ... (same pen and day, so step 06 gives them the same cow label until
relabelled).
"""

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .Detect import CocoCow
from .VideoTools import FormatTime, ProbeVideo


def _KeyFrames(VideoPath, EverySeconds, Width=1024):
    """``(TimeSeconds, BGR frame)`` every ``EverySeconds``, from key frames only."""
    Info = ProbeVideo(VideoPath)
    Height = int(round(Width * Info["Height"] / Info["Width"] / 2)) * 2
    Process = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-skip_frame", "nokey", "-i", str(VideoPath), "-an",
         "-vf", f"fps=1/{EverySeconds},scale={Width}:{Height}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        stdout=subprocess.PIPE)
    Size = Width * Height * 3
    try:
        Index = 0
        while True:
            Buffer = Process.stdout.read(Size)
            if len(Buffer) < Size:
                break
            yield Index * EverySeconds, np.frombuffer(Buffer, np.uint8).reshape(Height, Width, 3)
            Index += 1
    finally:
        Process.stdout.close()
        Process.kill()
        Process.wait()


def ScanVideo(VideoPath, EverySeconds=30, ModelName="yolo26n.pt", Confidence=0.25, ImageSize=1024) -> pd.DataFrame:
    """One row per looked-at frame: TimeSeconds, Cows (boxes >= 0.4), BestConfidence."""
    from ultralytics import YOLO

    Model, Rows, Started = YOLO(ModelName), [], time.time()
    for TimeSeconds, Frame in _KeyFrames(VideoPath, EverySeconds):
        R = Model.predict(Frame, classes=[CocoCow], conf=Confidence, imgsz=ImageSize, verbose=False)[0]
        Conf = R.boxes.conf.cpu().numpy() if R.boxes is not None and len(R.boxes) else np.zeros(0)
        Rows.append({"TimeSeconds": TimeSeconds, "Cows": int((Conf >= 0.4).sum()),
                     "BestConfidence": round(float(Conf.max()), 3) if len(Conf) else 0.0})
        if len(Rows) % 200 == 0:
            print(f"  {FormatTime(TimeSeconds)} scanned ({time.time() - Started:.0f} s)")
    return pd.DataFrame(Rows, columns=["TimeSeconds", "Cows", "BestConfidence"])


def PickWindows(Scan: pd.DataFrame, WindowMinutes=10, Windows=3, MinCowShare=0.8, MinConfidence=0.4) -> pd.DataFrame:
    """Up to ``Windows`` windows of ``WindowMinutes`` with a cow in view at least
    ``MinCowShare`` of the time, one from each equal part of the recording."""
    Columns = ["Window", "ClipStart", "ClipEnd", "StartSeconds", "EndSeconds", "CowShare", "MaxCows"]
    if Scan.empty:
        return pd.DataFrame(columns=Columns)
    Minute = (Scan.TimeSeconds // 60).astype(int)
    PerMinute = (Scan.BestConfidence >= MinConfidence).groupby(Minute).mean()
    CowsPerMinute = Scan.Cows.groupby(Minute).max()
    PerMinute = PerMinute.reindex(range(int(Minute.max()) + 1), fill_value=0.0)
    CowsPerMinute = CowsPerMinute.reindex(PerMinute.index, fill_value=0)
    Share = PerMinute.rolling(WindowMinutes).mean().shift(-(WindowMinutes - 1))   # share of the window starting here
    Total = len(PerMinute)
    Picked = []
    for Part in range(Windows):
        Lo, Hi = Part * Total // Windows, (Part + 1) * Total // Windows - WindowMinutes
        Candidates = Share.iloc[Lo:max(Hi, Lo) + 1].dropna()
        Candidates = Candidates[Candidates >= MinCowShare]
        if Candidates.empty:
            continue
        Start = int(Candidates.idxmax())
        if any(abs(Start - P) < WindowMinutes for P in Picked):
            continue
        Picked.append(Start)
    Rows = [{"Window": I + 1, "ClipStart": FormatTime(S * 60), "ClipEnd": FormatTime((S + WindowMinutes) * 60),
             "StartSeconds": S * 60, "EndSeconds": (S + WindowMinutes) * 60,
             "CowShare": round(float(Share.iloc[S]), 3), "MaxCows": int(CowsPerMinute.iloc[S:S + WindowMinutes].max())}
            for I, S in enumerate(sorted(Picked))]
    return pd.DataFrame(Rows, columns=Columns)


def _SaveTimeline(Scan, Windows, OutPath, Title, MinConfidence):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Hours = Scan.TimeSeconds / 3600
    Fig, Ax = plt.subplots(figsize=(12, 2.8))
    Ax.plot(Hours, Scan.Cows, lw=0.8, color="black", label="cows (conf >= 0.4)")
    Ax.fill_between(Hours, 0, (Scan.BestConfidence >= MinConfidence) * 0.3, color="tab:green", alpha=0.4, label="cow in view")
    for W in Windows.itertuples():
        Ax.axvspan(W.StartSeconds / 3600, W.EndSeconds / 3600, color="tab:orange", alpha=0.35)
        Ax.text(W.StartSeconds / 3600, Ax.get_ylim()[1] * 0.9, f"Win{W.Window}", fontsize=8)
    Ax.set_xlabel("hours into the video")
    Ax.set_ylabel("cows")
    Ax.set_title(Title, fontsize=10)
    Ax.legend(fontsize=7, loc="upper right")
    Fig.tight_layout()
    Fig.savefig(OutPath, dpi=110)
    plt.close(Fig)


def AutoWindows(VideoPath, ScanDir, EverySeconds=30, WindowMinutes=10, Windows=3, MinCowShare=0.8,
                MinConfidence=0.4, ImageSize=1024, Rescan=False) -> pd.DataFrame:
    """Scan (or reuse the scan) and pick windows; saves Scan.csv, Windows.csv,
    CowTimeline.png and Scan.json in ``ScanDir``."""
    ScanDir = Path(ScanDir)
    ScanDir.mkdir(parents=True, exist_ok=True)
    ScanSettings = {"Video": str(VideoPath), "EverySeconds": EverySeconds, "ImageSize": ImageSize}
    Info = ScanDir / "Scan.json"
    Same = Info.exists() and json.loads(Info.read_text()).get("Scan") == ScanSettings
    if Same and not Rescan and (ScanDir / "Scan.csv").exists():
        Scan = pd.read_csv(ScanDir / "Scan.csv")
        print(f"{Path(VideoPath).name}: reusing the scan ({len(Scan)} frames)")
    else:
        print(f"{Path(VideoPath).name}: scanning the whole video, one key frame every {EverySeconds} s ...")
        Started = time.time()
        Scan = ScanVideo(VideoPath, EverySeconds=EverySeconds, ImageSize=ImageSize)
        Scan.to_csv(ScanDir / "Scan.csv", index=False)
        print(f"  {FormatTime(Scan.TimeSeconds.max() if len(Scan) else 0)} of video scanned in {time.time() - Started:.0f} s")
    Picked = PickWindows(Scan, WindowMinutes, Windows, MinCowShare, MinConfidence)
    Picked.to_csv(ScanDir / "Windows.csv", index=False)
    Info.write_text(json.dumps({"Scan": ScanSettings, "Pick": {"WindowMinutes": WindowMinutes, "Windows": Windows,
                                                              "MinCowShare": MinCowShare, "MinConfidence": MinConfidence}}, indent=1))
    InView = float((Scan.BestConfidence >= MinConfidence).mean()) if len(Scan) else 0.0
    _SaveTimeline(Scan, Picked, ScanDir / "CowTimeline.png",
                  f"{Path(VideoPath).name}: cow in view {InView:.0%} of the time; {len(Picked)} window(s)", MinConfidence)
    print(f"  cow in view {InView:.0%} of the time; windows: "
          + (", ".join(f"Win{W.Window} {W.ClipStart}-{W.ClipEnd} ({W.CowShare:.0%}, max {W.MaxCows} cow(s))"
                       for W in Picked.itertuples()) or f"none with >= {MinCowShare:.0%} cow in view"))
    return Picked
