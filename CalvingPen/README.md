# CalvingPen

Pipeline that identifies individual cows (AnimalId) in calving-pen videos.
It is built one step at a time on a single reference recording
(`2026-09-26 00-00-00 Calving Pen 2 (2).avi`, 24 h) before it runs on
the rest.

## Where things live

| What | Where |
|---|---|
| Pipeline code (`CalvingPipeline/*.py`) | this repo |
| Notebooks | Shared Drive only. `*.ipynb` is git-ignored here because outputs show farm footage |
| Videos, clips, frames, outputs | Google Drive, never committed |

Each notebook clones this repo in its first cell and imports `CalvingPipeline`.

## Steps

| Step | Notebook | Status |
|---|---|---|
| 01 | `Step01PrepareReferenceClip.ipynb`: probe the video, cut `ReferenceClip.mp4` + `DevClip.mp4` (fixed names, overwritten on rerun) | done |
| 02 | `Step02DetectTrack.ipynb`: YOLO26n (COCO "cow") + ByteTrack on `DevClip.mp4` → `Tracks.csv`, `TrackSummary.csv`, `PreviewTracks.mp4` (GPU) | in review |
| 03 | Masks (SAM 2.1 Small) | |
| 04 | Embeddings (DINOv3 ViT-S/16) | |
| 05 | Reference gallery of labeled AnimalIds | |
| 06 | Re-ID prediction | |
| 07 | Evaluation | |

## Conventions

- PascalCase for files, folders, functions and variables.
- Never load a whole video into memory; seek or stream.
- Test on the short reference clip before full recordings.
- No videos, weights or tokens in git (Hugging Face token goes in Colab Secrets).
