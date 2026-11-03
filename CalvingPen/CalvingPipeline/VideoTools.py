"""Look inside long calving-pen recordings and cut short clips out of them.

Uses ffmpeg / ffprobe (preinstalled on Colab). Every function seeks to the
times it needs, so a 24 h AVI is never read end to end or held in memory.
"""

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np


def ParseTime(Time) -> float:
    """Seconds from a number or an "HH:MM:SS" / "MM:SS" string."""
    if isinstance(Time, (int, float)):
        return float(Time)
    Seconds = 0.0
    for Part in str(Time).split(":"):
        Seconds = Seconds * 60 + float(Part)
    return Seconds


def FormatTime(Seconds: float) -> str:
    Seconds = int(round(Seconds))
    return f"{Seconds // 3600:02d}:{Seconds % 3600 // 60:02d}:{Seconds % 60:02d}"


def ProbeVideo(VideoPath) -> dict:
    """Codec, resolution, FPS, duration and frame count of a video."""
    Result = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", "-select_streams", "v:0", str(VideoPath)],
        capture_output=True, text=True, check=True,
    )
    Info = json.loads(Result.stdout)
    Stream = Info["streams"][0]
    Format = Info["format"]

    Num, Den = Stream.get("avg_frame_rate", "0/1").split("/")
    Fps = float(Num) / float(Den) if float(Den) else 0.0
    DurationSeconds = float(Stream.get("duration") or Format.get("duration") or 0)
    FrameCount = int(Stream.get("nb_frames") or round(DurationSeconds * Fps))

    return {
        "Video": Path(VideoPath).name,
        "Container": Format.get("format_name"),
        "Codec": Stream.get("codec_name"),
        "PixelFormat": Stream.get("pix_fmt"),
        "Width": Stream.get("width"),
        "Height": Stream.get("height"),
        "Fps": round(Fps, 3),
        "FrameCount": FrameCount,
        "DurationSeconds": round(DurationSeconds, 2),
        "Duration": FormatTime(DurationSeconds),
        "SizeMb": round(int(Format.get("size", 0)) / 2**20, 1),
    }


def IterFrames(VideoPath, Wanted):
    """Yield ``(FrameIndex, BGR frame)`` for the frame indices in ``Wanted``,
    reading the video once in order and stopping after the last one."""
    Wanted = set(int(I) for I in Wanted)
    if not Wanted:
        return
    Last = max(Wanted)
    Cap = cv2.VideoCapture(str(VideoPath))
    try:
        for FrameIndex in range(Last + 1):
            if not Cap.grab():
                break
            if FrameIndex in Wanted:
                Ok, Frame = Cap.retrieve()
                if Ok:
                    yield FrameIndex, Frame
    finally:
        Cap.release()


def GrabFrameAt(VideoPath, Time, Width=None) -> np.ndarray:
    """One BGR frame at (about) ``Time``; ffmpeg snaps to a nearby keyframe."""
    Scale = ["-vf", f"scale={Width}:-2"] if Width else []
    Result = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", str(ParseTime(Time)), "-i", str(VideoPath),
         "-frames:v", "1", *Scale, "-f", "image2pipe", "-c:v", "png", "-"],
        capture_output=True, check=True,
    )
    Frame = cv2.imdecode(np.frombuffer(Result.stdout, np.uint8), cv2.IMREAD_COLOR)
    if Frame is None:
        raise ValueError(f"no frame at {FormatTime(ParseTime(Time))} in {VideoPath}")
    return Frame


def SaveTimelineSheet(VideoPath, OutPath, EveryMinutes=15, ThumbWidth=320, Columns=8) -> Path:
    """A grid of thumbnails, one every ``EveryMinutes``, each stamped with its
    time in the video. Use it to find when the cow is in the pen / calving."""
    DurationSeconds = ProbeVideo(VideoPath)["DurationSeconds"]
    Times = np.arange(0, DurationSeconds, EveryMinutes * 60)

    Thumbs = []
    for Time in Times:
        try:
            Thumb = GrabFrameAt(VideoPath, Time, Width=ThumbWidth)
        except (subprocess.CalledProcessError, ValueError):
            Thumb = None
        if Thumb is None:
            Height = Thumbs[0].shape[0] if Thumbs else ThumbWidth * 9 // 16
            Thumb = np.zeros((Height, ThumbWidth, 3), np.uint8)
        Label = FormatTime(Time)
        cv2.putText(Thumb, Label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(Thumb, Label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        Thumbs.append(Thumb)

    Height = Thumbs[0].shape[0]
    Thumbs = [cv2.resize(T, (ThumbWidth, Height)) for T in Thumbs]
    Thumbs += [np.zeros_like(Thumbs[0])] * (-len(Thumbs) % Columns)
    Rows = [np.hstack(Thumbs[I:I + Columns]) for I in range(0, len(Thumbs), Columns)]

    OutPath = Path(OutPath)
    OutPath.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OutPath), np.vstack(Rows))
    return OutPath


def CutClip(VideoPath, StartTime, DurationSeconds, OutPath, ReEncode=True) -> Path:
    """Cut ``DurationSeconds`` starting at ``StartTime`` into a new file.

    ``ReEncode=True`` writes H.264 MP4 (frame-accurate, plays everywhere,
    small); ``False`` copies the original stream (fast, but the cut snaps
    to keyframes and keeps the original codec).
    """
    OutPath = Path(OutPath)
    OutPath.parent.mkdir(parents=True, exist_ok=True)
    Codec = (["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"]
             if ReEncode else ["-c:v", "copy"])
    # AVI streams often lack timestamps; regenerate them so the MP4 plays and seeks.
    Pts = [] if ReEncode else ["-fflags", "+genpts"]
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", *Pts, "-ss", str(ParseTime(StartTime)), "-i", str(VideoPath),
         "-t", str(ParseTime(DurationSeconds)), *Codec, "-an", str(OutPath)],
        check=True,
    )
    return OutPath


def PrepareClips(VideoPath, OutputDir, ClipStart, ClipEnd=None, DevStart=None, DevSeconds=300) -> dict:
    """Probe the video and cut its clips into ``OutputDir`` under fixed names,
    overwriting any earlier run for this video:

    - ``DevClip.mp4``: ``DevSeconds`` from ``DevStart``, re-encoded; cut first
      because the next step is built on it
    - ``ReferenceClip.mp4``: ``ClipStart`` to ``ClipEnd`` (None = end of video),
      a stream copy, so an hour of video takes minutes; it may start up to one
      keyframe before ``ClipStart``
    - ``Probe.json``: video and clip info plus the settings used, rewritten
      after each clip so a stopped run still records what finished
    """
    OutputDir = Path(OutputDir)
    OutputDir.mkdir(parents=True, exist_ok=True)
    Video = ProbeVideo(VideoPath)

    Start = ParseTime(ClipStart)
    End = Video["DurationSeconds"] if ClipEnd is None else min(ParseTime(ClipEnd), Video["DurationSeconds"])
    if End <= Start:
        raise ValueError(f"ClipStart {FormatTime(Start)} is not before the end {FormatTime(End)}")

    Summary = {"Video": Video, "ClipStart": FormatTime(Start), "ClipEnd": FormatTime(End)}
    SummaryPath = OutputDir / "Probe.json"

    if DevStart is not None:
        Summary["DevStart"] = FormatTime(ParseTime(DevStart))
        print(f"DevClip {DevSeconds} s from {Summary['DevStart']} ...")
        Summary["DevClip"] = ProbeVideo(CutClip(VideoPath, DevStart, DevSeconds, OutputDir / "DevClip.mp4"))
        SummaryPath.write_text(json.dumps(Summary, indent=2))

    print(f"ReferenceClip {FormatTime(Start)} to {FormatTime(End)} (stream copy) ...")
    Summary["ReferenceClip"] = ProbeVideo(
        CutClip(VideoPath, Start, End - Start, OutputDir / "ReferenceClip.mp4", ReEncode=False))
    SummaryPath.write_text(json.dumps(Summary, indent=2))
    return Summary
