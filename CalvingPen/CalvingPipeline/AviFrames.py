"""Read AVI files whose index and timestamps are broken (the parlor recorder).

The entry-parlor recorder writes an AVI where every 1/150 s gets a video
chunk, but a chunk is **empty** (8 bytes, no picture) whenever the picture did
not change. The first real frame is followed by a long run of empty chunks,
and the file index is unusable, so ffmpeg and OpenCV stop after frame 1.

``AviChunks`` walks the file chunk by chunk (RIFF / LIST movi / 00dc ...),
counting one tick per video chunk, empty or not. ``RebuildAvi`` keeps the real
(non-empty) frames with their time (tick / ticks per second), decodes them as
plain H.264, and writes a normal clip at ``OutFps`` for ``Start``..``Start+Seconds``,
holding each picture until the next real frame. ``AviStats`` tells how many
ticks and real frames a file has and when they occur.
"""

import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def AviChunks(VideoPath):
    """Yield ``(Tick, Offset, Size, IsKey)`` for every video chunk in file order.
    ``IsKey``: the chunk holds an H.264 SPS or IDR (decoding can start there)."""
    Tick = 0
    with open(VideoPath, "rb") as F:
        F.seek(0, 2)
        FileSize = F.tell()
        F.seek(0)

        def Walk(End):
            nonlocal Tick
            while F.tell() + 8 <= End:
                Header = F.read(8)
                if len(Header) < 8:
                    return
                Id, Size = Header[:4], struct.unpack("<I", Header[4:])[0]
                Start = F.tell()
                if Id in (b"RIFF", b"LIST"):
                    Kind = F.read(4)
                    if Kind in (b"movi", b"rec ", b"AVI ", b"AVIX"):
                        yield from Walk(min(Start + Size, FileSize))
                    F.seek(Start + Size + (Size & 1))
                    continue
                if len(Id) == 4 and Id[2:4] in (b"dc", b"db") and Id[:2].isdigit():
                    IsKey = False
                    if Size:
                        Head = F.read(min(Size, 64))
                        IsKey = any((Head[I + 4] & 0x1F) in (5, 7)
                                    for I in range(len(Head) - 4) if Head[I:I + 4] == b"\x00\x00\x00\x01")
                    yield Tick, Start, Size, IsKey
                    Tick += 1
                if Start + Size > FileSize:          # cut-off file (e.g. a sample): stop cleanly
                    return
                F.seek(Start + Size + (Size & 1))

        yield from Walk(FileSize)


def AviStats(VideoPath, TicksPerSecond=150.0) -> dict:
    """Ticks, real frames and keyframes of a file, and when the real frames are."""
    Real = [(T, K) for T, _, S, K in AviChunks(VideoPath) if S]
    Ticks = Real[-1][0] + 1 if Real else 0
    Times = np.array([T for T, _ in Real]) / TicksPerSecond
    return {"RealFrames": len(Real), "KeyFrames": int(sum(K for _, K in Real)),
            "FirstRealSeconds": round(float(Times[0]), 2) if len(Times) else None,
            "LastRealSeconds": round(float(Times[-1]), 2) if len(Times) else None,
            "MedianGapSeconds": round(float(np.median(np.diff(Times))), 3) if len(Times) > 1 else None}


def RebuildAvi(VideoPath, OutPath, StartSeconds=0.0, Seconds=None, TicksPerSecond=150.0, OutFps=30,
               Width=None, Height=None) -> dict:
    """Write ``StartSeconds``..``+Seconds`` (None = to the end) as an H.264 MP4 at
    ``OutFps`` from the real frames, holding each until the next one."""
    OutPath = Path(OutPath)
    OutPath.parent.mkdir(parents=True, exist_ok=True)
    End = None if Seconds is None else StartSeconds + Seconds
    # 1) the real frames needed: from the last keyframe before the window to its end
    Chunks, LastKey = [], None
    for Tick, Offset, Size, IsKey in AviChunks(VideoPath):
        T = Tick / TicksPerSecond
        if End is not None and T > End:
            break
        if not Size:
            continue
        if T < StartSeconds:
            if IsKey:
                Chunks, LastKey = [(T, Offset, Size)], T
            elif LastKey is not None:
                Chunks.append((T, Offset, Size))
            continue
        if LastKey is None and not Chunks and not IsKey:
            continue                                   # cannot decode before the first keyframe
        Chunks.append((T, Offset, Size))
        LastKey = T if LastKey is None else LastKey
    if not Chunks:
        raise RuntimeError(f"No real frames in {VideoPath} between {StartSeconds} s and {End} s")
    # 2) their H.264 data as one elementary stream, decoded in order (one picture per chunk)
    with tempfile.TemporaryDirectory() as Tmp:
        Raw = Path(Tmp) / "Frames.h264"
        with open(VideoPath, "rb") as F, open(Raw, "wb") as G:
            for _, Offset, Size in Chunks:
                F.seek(Offset)
                G.write(F.read(Size))
        if not (Width and Height):
            Probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                    "stream=width,height", "-of", "csv=p=0", str(Raw)], capture_output=True, text=True)
            Width, Height = map(int, Probe.stdout.strip().split(",")[:2])
        FrameBytes = Width * Height * 3
        Decoder = subprocess.Popen(["ffmpeg", "-v", "error", "-f", "h264", "-framerate", "30", "-i", str(Raw),
                                    "-r", "30", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE,
                                   stderr=subprocess.DEVNULL)   # it is stopped early on purpose: no "broken pipe" noise
        Encoder = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                                    "-s", f"{Width}x{Height}", "-r", str(OutFps), "-i", "-", "-c:v", "libx264",
                                    "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-an", str(OutPath)],
                                   stdin=subprocess.PIPE)
        # 3) hold each decoded picture until the next real frame's time
        Times = [T for T, _, _ in Chunks]
        LastTime = End if End is not None else Times[-1] + 1 / OutFps
        Written, Current, Index, J = 0, None, 0, 0
        try:
            while StartSeconds + J / OutFps < LastTime:
                Now = StartSeconds + J / OutFps
                while Index < len(Times) and Times[Index] <= Now:     # decode every real frame up to now
                    Buffer = Decoder.stdout.read(FrameBytes)
                    if len(Buffer) < FrameBytes:
                        Index = len(Times)
                        break
                    Current, Index = Buffer, Index + 1
                if Current is not None:                               # hold the latest picture
                    Encoder.stdin.write(Current)
                    Written += 1
                J += 1
        finally:
            Decoder.stdout.close()
            Decoder.kill()
            Encoder.stdin.close()
            Encoder.wait()
    Real = sum(1 for T in Times if T >= StartSeconds)
    return {"RealFrames": Real, "WrittenFrames": Written, "Seconds": round(Written / OutFps, 1)}


def AviActivity(VideoPath, TicksPerSecond=150.0, BinSeconds=60) -> "pd.DataFrame":
    """Real (non-empty) frames per minute of the recording: where the picture
    actually changes (cows moving). Empty stretches are frozen pictures."""
    import pandas as pd
    from .VideoTools import FormatTime

    Times = np.array([T for T, _, S, _ in AviChunks(VideoPath) if S]) / TicksPerSecond
    if not len(Times):
        return pd.DataFrame(columns=["Minute", "Time", "RealFrames"])
    Counts = np.bincount((Times // BinSeconds).astype(int))
    return pd.DataFrame({"Minute": np.arange(len(Counts)), "Time": [FormatTime(M * BinSeconds) for M in range(len(Counts))],
                         "RealFrames": Counts})


def BusiestWindow(Activity, Minutes=5) -> tuple:
    """(StartSeconds, EndSeconds) of the ``Minutes`` with the most real frames."""
    Counts = Activity.RealFrames.to_numpy()
    if len(Counts) <= Minutes:
        return 0, len(Counts) * 60
    Sums = np.convolve(Counts, np.ones(Minutes, int), "valid")
    Start = int(np.argmax(Sums))
    return Start * 60, (Start + Minutes) * 60
