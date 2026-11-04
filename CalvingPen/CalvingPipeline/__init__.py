"""Calving-pen cow re-identification pipeline. One module per step:

| Module        | Step | Does                                              | Model / tool                              |
|---------------|------|---------------------------------------------------|-------------------------------------------|
| Paths.py      | all  | Shared Drive locations of videos and outputs       |                                           |
| VideoTools.py | 01   | probe a video, cut the clip, read frames in order | ffmpeg / ffprobe, OpenCV                  |
| DetectTrack.py| 02   | box every cow, link boxes into tracks             | YOLO26n (COCO "cow" class) + ByteTrack    |
| Sam2Track.py  | 02c  | experiment: follow the cow's mask over time       | YOLO26n seeds + SAM 2.1 Small video mode  |
| CropCows.py   | 03   | clean tracks, pick frames, cut + mask the cow     | SAM 2.1 Small                             |
| Embed.py      | 04   | one 384-number fingerprint per crop               | DINOv3 ViT-S/16 (fallback DINOv2-small)   |
| Pool.py       | 05   | pool all videos, label tracks, self-check         | (reuses step 04 fingerprints)             |

Weights download automatically the first time (Ultralytics for YOLO / SAM,
Hugging Face for DINO). None are trained on our cows. Terms (track, crop,
mask, fingerprint) and results are explained in CalvingPen/README.md.

Only the light step-01 tools are imported here; the other modules are
imported directly (``from CalvingPipeline.DetectTrack import ...``) so step 01
runs without the GPU libraries.
"""

from .VideoTools import CutClip, FormatTime, GrabFrameAt, ParseTime, PrepareClips, ProbeVideo, SaveTimelineSheet
