"""Read a video at a fixed rate with times we trust (stage 1 of the re-ID design).

The calving and parlor recorders write AVIs with one chunk per 1/150 s; a chunk
is empty when the picture did not change, and the header length is often fake
(24-80 h for a 4 h recording). Container timestamps are therefore never used:
for an AVI we walk the chunks ourselves (``AviFrames.AviChunks``), give every
real (non-empty) chunk the time ``tick / NativeFps``, decode the H.264 data with
PyAV, and hold each picture until the next real one. The file ends where its
last real chunk is, so an empty or broken tail is never read as video.

Other files (.mp4 cut by ffmpeg) have sane timestamps and are read normally.

``SampleFrames`` yields ``(Tick, TimeSeconds, Frame)`` every ``1 / TargetFps``
seconds; ``Tick`` names the real picture shown (the same picture can be
yielded several times when nothing changed).
"""

import shutil
from pathlib import Path

import numpy as np

from ..AviFrames import AviChunks


def LocalCopy(VideoPath, LocalDir="/content/ReidVideos") -> Path:
    """Copy the video to the Colab disk once (walking millions of chunks over
    the Drive mount is very slow); reused while the size matches."""
    VideoPath, LocalDir = Path(VideoPath), Path(LocalDir)
    if not LocalDir.parent.exists():          # not on Colab: read in place
        return VideoPath
    LocalDir.mkdir(parents=True, exist_ok=True)
    Local = LocalDir / VideoPath.name
    if not Local.exists() or Local.stat().st_size != VideoPath.stat().st_size:
        print(f"Copying {VideoPath.name} ({VideoPath.stat().st_size / 2**30:.1f} GB) to the Colab disk ...")
        shutil.copyfile(VideoPath, Local)
    return Local


def HeaderFps(VideoPath) -> float:
    from ..VideoTools import ProbeVideo
    return float(ProbeVideo(VideoPath)["Fps"] or 30.0)


def _AviPictures(VideoPath, NativeFps, StartSeconds, EndSeconds):
    """``(Tick, TimeSeconds, av.VideoFrame)`` for every real picture, in order,
    decoding from the last keyframe before ``StartSeconds``."""
    import av

    Wanted, LastKey = [], None
    for Tick, Offset, Size, IsKey in AviChunks(VideoPath):
        if not Size:
            continue
        T = Tick / NativeFps
        if EndSeconds is not None and T > EndSeconds:
            break
        if T < StartSeconds:
            if IsKey:
                Wanted, LastKey = [(Tick, Offset, Size)], Tick
            elif LastKey is not None:
                Wanted.append((Tick, Offset, Size))
            continue
        if not Wanted and not IsKey:
            continue                                  # nothing decodable before the first keyframe
        Wanted.append((Tick, Offset, Size))
    Codec = av.CodecContext.create("h264", "r")
    Codec.thread_type = "AUTO"
    with open(VideoPath, "rb") as F:
        for Tick, Offset, Size in Wanted:
            F.seek(Offset)
            Packet = av.Packet(F.read(Size))
            Packet.pts = Tick                         # the decoder passes it through: the frame keeps its tick
            try:
                Decoded = Codec.decode(Packet)
            except av.error.InvalidDataError:
                continue
            for Frame in Decoded:
                yield Frame.pts, Frame.pts / NativeFps, Frame
        for Frame in Codec.decode(None):
            yield Frame.pts, Frame.pts / NativeFps, Frame


def _OtherPictures(VideoPath, StartSeconds, EndSeconds):
    import av

    with av.open(str(VideoPath)) as Container:
        Stream = Container.streams.video[0]
        Stream.thread_type = "AUTO"
        for Index, Frame in enumerate(Container.decode(Stream)):
            T = float(Frame.time) if Frame.time is not None else Index / float(Stream.average_rate or 30)
            if EndSeconds is not None and T > EndSeconds:
                return
            yield Index, T, Frame


def SampleFrames(VideoPath, TargetFps=5, StartSeconds=0.0, EndSeconds=None, NativeFps=None, Info=None):
    """Yield ``(Tick, TimeSeconds, BGR frame)`` at ``StartSeconds + k / TargetFps``
    up to the last real picture (or ``EndSeconds``). ``NativeFps``: ticks per
    second of an AVI (default: the rate its header claims, 150 for our cameras).
    ``Info`` (a dict) receives ``LastRealSeconds`` and ``RealPictures``."""
    VideoPath = Path(VideoPath)
    IsAvi = VideoPath.suffix.lower() == ".avi"
    NativeFps = float(NativeFps or (HeaderFps(VideoPath) if IsAvi else 30.0))
    Pictures = (_AviPictures(VideoPath, NativeFps, StartSeconds, EndSeconds) if IsAvi
                else _OtherPictures(VideoPath, StartSeconds, EndSeconds))
    Step, K = 1.0 / TargetFps, 0
    Current, CurrentArray, Count, LastT = None, None, 0, None
    for Tick, T, Frame in Pictures:
        Count, LastT = Count + 1, T
        while Current is not None and StartSeconds + K * Step < T:
            TargetT = StartSeconds + K * Step
            if EndSeconds is not None and TargetT > EndSeconds:
                break
            if CurrentArray is None:
                CurrentArray = Current[1].to_ndarray(format="bgr24")
            yield Current[0], round(TargetT, 3), CurrentArray
            K += 1
        Current, CurrentArray = (Tick, Frame, T), None      # pictures come in time order
    # the last picture: shown once more at the next sampling time, then the recording ends
    if Current is not None and StartSeconds + K * Step <= Current[2] + Step and (
            EndSeconds is None or StartSeconds + K * Step <= EndSeconds):
        yield Current[0], round(StartSeconds + K * Step, 3), (
            CurrentArray if CurrentArray is not None else Current[1].to_ndarray(format="bgr24"))
    if Info is not None:
        Info.update({"LastRealSeconds": None if LastT is None else round(LastT, 3), "RealPictures": Count,
                     "NativeFps": NativeFps})
