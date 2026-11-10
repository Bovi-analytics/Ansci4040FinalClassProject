"""Calving-pen cow re-identification pipeline. One module per step; each step
is one function, so it can be run for any video (any cow) from a notebook:

| Step | Module        | Function       | Does                                           | Model / tool                     |
|------|---------------|----------------|------------------------------------------------|----------------------------------|
| all  | Paths.py      |                | Shared Drive locations of videos and outputs   |                                  |
| 01   | VideoTools.py | PrepareClips   | probe a video, cut the window with the cow     | ffmpeg / ffprobe                 |
| 02   | Detect.py     | DetectCows     | box every cow in every sampled frame           | YOLO26n (COCO "cow" class)       |
| 03   | Track.py      | TrackCows      | follow each cow's outline through the clip     | SAM 2.1 Small, video mode        |
| 04   | Crop.py       | CropCows       | clean tracks, pick frames, cut + mask the cow  | SAM 2.1 Small, image mode        |
| 05   | Embed.py      | EmbedCrops     | one 384-number fingerprint per crop            | DINOv3 ViT-S/16 (fallback DINOv2)|
| 06   | Pool.py       | BuildPool      | pool all videos, label tracks, self-check      | (reuses step 05 fingerprints)    |
| -    | Baselines/ByteTrack.py | DetectAndTrack | old tracker, only for comparison      | YOLO26n + ByteTrack              |

Weights download automatically the first time (Ultralytics for YOLO / SAM,
Hugging Face for DINO). None are trained on our cows. Terms (track, crop,
mask, fingerprint) and results are explained in CalvingPen/README.md.

Only the light step-01 tools are imported here; the other modules are
imported directly (``from CalvingPipeline.Detect import DetectCows``) so step
01 runs without the GPU libraries.
"""

from .VideoTools import CutClip, FormatTime, GrabFrameAt, ParseTime, PrepareClips, ProbeVideo, SaveTimelineSheet
