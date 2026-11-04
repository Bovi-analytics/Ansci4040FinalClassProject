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

Outputs go to the group's Shared Drive, `ANSC 4040 Group Project/CalvingPenPipelineOutputs/`
(set once in `CalvingPipeline/Paths.py`), one folder per video, one subfolder per step.
The folder is shared, so one person runs a given step at a time:

```
<video name>/
├── ReferenceClip.mp4, Probe.json          Step 01
├── Step02aTest/                           Step 02a: first minute, 15 fps
│   └── Tracks.csv, TrackSummary.csv, PreviewTracks.mp4
└── Step02bFull/                           Step 02b: whole clip, 5 fps
    └── Tracks.csv, TrackSummary.csv, PreviewTracks.mp4 (4x speed)
├── Step03aTest/, Step03bFull/             Step 03: KeptTracks.csv, Crops.csv, Crops/, Sheets/
├── Step04aTest/, Step04bFull/             Step 04: Embeddings.npy, EmbeddingIndex.csv, TrackSimilarity.csv/.png
```

## Steps

| Step | Notebook | Status |
|---|---|---|
| 01 | `Step01PrepareReferenceClip.ipynb`: probe the video, cut `ReferenceClip.mp4` (stream copy) | done |
| 02a | `Step02aDetectTrackTest.ipynb`: YOLO26n (COCO "cow") + ByteTrack on the first minute, real-time preview (~1 min, GPU) | done |
| 02b | `Step02bDetectTrackFull.ipynb`: same on the whole clip at 5 fps, 4x preview (~20 min, GPU) | done |
| 03a | `Step03aCropCowsTest.ipynb`: crops + SAM 2.1 masks for tracks in the first 2 min, 8 per track (GPU) | in review |
| 03b | `Step03bCropCowsFull.ipynb`: same for the whole hour, 24 per track; drops short / unsure tracks | in review |
| 04a | `Step04aEmbedTest.ipynb`: DINOv3 ViT-S/16 fingerprints (384-d) of the 03a crops; DINOv2-small fallback without HF access (GPU) | in review |
| 04b | `Step04bEmbedFull.ipynb`: same for all 03b crops → `Embeddings.npy`, `EmbeddingIndex.csv`, `TrackSimilarity.png` | in review |
| 05 | Reference gallery of labeled AnimalIds | |
| 06 | Re-ID prediction | |
| 07 | Evaluation | |

## Conventions

- PascalCase for files, folders, functions and variables.
- Never load a whole video into memory; seek or stream.
- Test on a short piece (Step 02a style) before full recordings.
- Every step overwrites its own output folder; edit the settings cell and rerun instead of adding cells.
- No videos, weights or tokens in git. The Hugging Face token goes in Colab Secrets as `HF_TOKEN`.

## Known issues

- A calf touching the cow (e.g. being licked) can end up inside her SAM mask (Step 03, track 198 of the reference clip).
- The reference clip has one cow, so Re-ID needs crops from other videos (or the milking-parlor group) to tell cows apart.
- The cow's AnimalId is unknown; she is labelled locally (e.g. `Pen2_20260926_Cow01`) until it can be linked.
