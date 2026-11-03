# CalvingPen

Pipeline that identifies individual cows (AnimalId) in calving-pen videos.
It is built one step at a time on a single reference recording
(`2026-09-26 00-00-00 Calving Pen 2.avi`, 4 h 37 min; the window
03:35:11 to the end has the calving) before it runs on the rest.

## Where things live

| What | Where |
|---|---|
| Pipeline code (`CalvingPipeline/*.py`) | this repo |
| Notebooks (`Step*.ipynb`) | repo root, saved **without outputs**. Open them in Colab from GitHub; save your run with File > Save a copy in Drive, never back to GitHub |
| Videos, clips, previews, CSVs | Google Drive, never committed |

Each notebook clones this repo in its first cell and imports `CalvingPipeline`.

Drive layout (`MyDrive/CalvingPenPipeline/`), one folder per video, one subfolder per step:

```
<video name>/
├── ReferenceClip.mp4, Probe.json          Step 01
├── Step02aTest/                           Step 02a: first minute, 15 fps
│   └── Tracks.csv, TrackSummary.csv, PreviewTracks.mp4
└── Step02bFull/                           Step 02b: whole clip, 5 fps
    └── Tracks.csv, TrackSummary.csv, PreviewTracks.mp4 (4x speed)
```

## Steps

| Step | Notebook | Status |
|---|---|---|
| 01 | `Step01PrepareReferenceClip.ipynb`: probe the video, cut `ReferenceClip.mp4` (stream copy) | done |
| 02a | `Step02aDetectTrackTest.ipynb`: YOLO26n (COCO "cow") + ByteTrack on the first minute, real-time preview (~1 min, GPU) | done |
| 02b | `Step02bDetectTrackFull.ipynb`: same on the whole clip at 5 fps, 4x preview (~20 min, GPU) | done |
| 03 | Crops + masks (SAM 2.1 Small); drop short / low-confidence tracks; separate the calf | next |
| 04 | Embeddings (DINOv3 ViT-S/16) | |
| 05 | Reference gallery of labeled AnimalIds | |
| 06 | Re-ID prediction | |
| 07 | Evaluation | |

## Conventions

- PascalCase for files, folders, functions and variables.
- Never load a whole video into memory; seek or stream.
- Test on a short piece (Step 02a style) before full recordings.
- Every step overwrites its own output folder; edit the settings cell and rerun instead of adding cells.
- No videos, weights or tokens in git (Hugging Face token goes in Colab Secrets).
