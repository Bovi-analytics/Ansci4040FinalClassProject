# CalvingPen

Pipeline that identifies individual cows in calving-pen videos. It turns a
long recording into short, clean pictures of each cow and a numeric
"fingerprint" per picture, so the same cow can later be recognised in other
calving videos and in the milking-parlor videos (the other group's part).

It is built one step at a time on one reference recording before it runs on
the rest: `2026-09-26 00-00-00 Calving Pen 2.avi` (4 h 37 min), using the
window **03:35:11 to the end**, which contains one whole calving.

## How it works, in plain words

```
AVI video ─► 01 cut the window ─► 02 find + follow the cow ─► 03 cut her out ─► 04 fingerprint ─► 05 pool ─► 06 recognise
             ReferenceClip.mp4     boxes + track IDs            masked crops       384 numbers      all videos  which cow?
```

| Step | What it does | Model / tool | Why |
|---|---|---|---|
| 01 Prepare clip | Reads the video's details and copies the chosen time window into `ReferenceClip.mp4` | ffmpeg (stream copy, no re-encoding) | The full recording is hours long; we only need the part with the cow |
| 02 Detect + track | In every processed frame, draws a **box** around each cow; links boxes across frames into **tracks** | **YOLO26n** (Ultralytics, trained on the COCO dataset; we keep only its "cow" class) + **ByteTrack** tracker | Finds where the cows are and which box belongs to which one from one moment to the next |
| 03 Crop + mask | Drops duplicate and too-short / unsure tracks; picks ~24 frames spread over each track; cuts the cow out and greys out everything that is not her | **SAM 2.1 Small** (Meta's Segment Anything 2.1, given the box as a hint) | Clean pictures of only the cow, from many angles |
| 04 Fingerprint | Turns each picture into 384 numbers describing her coat pattern and shape | **DINOv3 ViT-S/16** (Meta, gated on Hugging Face; open **DINOv2-small** as fallback) | Pictures of the same cow get similar numbers; that is what recognition compares |
| 05 Pool | Collects the fingerprints of all videos, labels every track with its cow (`Identities.csv` per video), and checks how well each crop finds its own cow in *other* tracks | none (reuses step 04) | The reference set to recognise cows against |
| 06 Re-ID | For a new picture, finds the most similar pictures in the pool; "unknown" if nothing is close enough | (planned) | Answers "which cow is this?" |
| 07 Evaluation | Accuracy, unknown-cow rejection, speed | (planned) | |

None of the models were trained on our cows; all are used as downloaded.

## Glossary

| Term | Meaning |
|---|---|
| **Detection / box** | A rectangle the detector draws around something it thinks is a cow, with a **confidence** from 0 to 1 |
| **Track / TrackId** | One stretch of time in which the tracker followed the same box from frame to frame. A TrackId is **not a cow**: when the detector loses her for more than 10 s (lying, hidden), the next box gets a new TrackId. One cow can have many tracks |
| **Crop** | A picture cut out of a frame around one box |
| **Mask** | The outline of the animal inside the crop (from SAM); everything outside it is painted grey |
| **Embedding / fingerprint** | The 384 numbers a model computes from a crop |
| **Cosine similarity** | How alike two fingerprints are: about 1 = look the same, lower = look different |
| **Pool / gallery** | Fingerprints of cows whose identity we know (or have labelled locally) |
| **IdentityId** | The cow label a track gets in the pool, e.g. `Pen2_20260926_Cow01` (pen, date, cow number); set per track in the video's `Identities.csv` |
| **Self-check** | Hide one track, find the closest crops among all the others: does it land on the same cow? (leave-one-track-out) |
| **Re-ID** | Re-identification: deciding which known cow a new crop shows |
| **AnimalId** | The farm's ID for a cow. Unknown for the reference cow, so she is labelled locally as `Pen2_20260926_Cow01` |

## What we found on the reference clip

| Step | Result |
|---|---|
| 01 | The AVI header says 150 fps, but the real rate is **30 fps** (the recorder writes ~5 index entries per frame). The copied clip is 1:02:01, 1920×1080, 857 MB. About 93 % of consecutive frames are identical, so the camera updates the picture only a few times a second |
| 02 | One cow in the pen, but **47 tracks**: the detector loses her while she lies down or strains (630–2500 s), and sometimes puts a second, weaker box on her head or rear. 18,603 frames processed at 5 fps in ~20 min on a T4 |
| 03 | 167 duplicate boxes removed; **14 of 40 tracks kept** (the dropped ones are 1–18-frame fragments); **306 crops, all the same cow**. SAM cuts her out cleanly. When she licks the calf, the calf can end up inside her mask (track 198). The calf itself was never kept as a separate track |
| 04 | DINOv3 groups her crops by **pose**, not by track: lying tracks (5, 27, 37, 57, 125, 156, 169) look alike (0.76–0.86), standing / rear / after-birth tracks (54, 186, 198, 234) look alike (0.74–0.84), but lying vs. standing drops to 0.57–0.62 although it is the same cow. So Re-ID must match each crop against her **closest** crops, not an average. Different cows need to score below ~0.55 for this to work; that can only be checked with a second cow |

Track timeline (clip seconds; add 03:35:11 for the time in the video):
T2 3–48 standing · T5 51–401 lying · T27 400–481 lying · T37 475–513 lying · T40 513–630 standing ·
T54 1081–1097 standing, rear · T57 1117–1194 lying · T125 1982–2003 lying · T156 2199–2263 lying ·
T169 2302–2409 lying · T186 2477–2513 standing, just after the birth · T198 2516–3256 licking the calf ·
T234 3265–3623 standing, rear · T253 3627–3714 standing, close to the camera.
The calf appears to be born around 04:15–04:16 video time.

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
├── Step02bFull/                           Step 02b: whole clip, 5 fps
│   └── Tracks.csv, TrackSummary.csv, PreviewTracks.mp4 (4x speed)
├── Step03aTest/, Step03bFull/             Step 03: KeptTracks.csv, Crops.csv, Crops/, Sheets/
├── Step04aTest/, Step04bFull/             Step 04: Embeddings.npy, EmbeddingIndex.csv, TrackSimilarity.csv/.png
└── Identities.csv                         Step 05 input: which cow each track is (edit by hand; never overwritten)
Pool/, PoolTest/                           Step 05 (next to the video folders): PoolEmbeddings.npy, PoolIndex.csv,
                                           PoolSummary.csv, SelfCheck.csv/.png
```

## Notebooks

"a" notebooks are quick tests on a small piece; "b" notebooks run the whole clip. Run "a" first after changing settings.

| Step | Notebook | Runtime | Status |
|---|---|---|---|
| 01 | `Step01PrepareReferenceClip.ipynb` | CPU, a few min | done |
| 02a | `Step02aDetectTrackTest.ipynb`: first minute, 15 fps, real-time preview | T4, ~1 min | done |
| 02b | `Step02bDetectTrackFull.ipynb`: whole clip, 5 fps, 4x preview | T4, ~20 min | done |
| 03a | `Step03aCropCowsTest.ipynb`: tracks in the first 2 min, 8 crops each | T4, ~1 min | done |
| 03b | `Step03bCropCowsFull.ipynb`: all tracks, 24 crops each | T4, ~7 min | done |
| 04a | `Step04aEmbedTest.ipynb`: the 03a crops | T4, <1 min | done |
| 04b | `Step04bEmbedFull.ipynb`: the 03b crops | T4, ~2 min | done |
| 05a | `Step05aPoolTest.ipynb`: pools the 04a test fingerprints → `PoolTest/` | CPU, seconds | in review |
| 05b | `Step05bPoolFull.ipynb`: pools the 04b fingerprints of every video → `Pool/` | CPU, seconds | in review |
| 06 | Re-ID prediction | | |
| 07 | Evaluation | | |

## Conventions

- PascalCase for files, folders, functions and variables.
- Never load a whole video into memory; seek or stream.
- Test on a short piece ("a" notebooks) before full recordings.
- Every step overwrites its own output folder; edit the settings cell and rerun instead of adding cells.
- No videos, weights or tokens in git. The Hugging Face token goes in Colab Secrets as `HF_TOKEN`.

## Known issues

- One cow becomes many tracks (see the glossary); Re-ID, not the tracker, decides which tracks are the same cow.
- A calf touching the cow (e.g. being licked) can end up inside her SAM mask (Step 03, track 198).
- The fingerprint depends strongly on pose (standing vs. lying); see the step 04 result above.
- The reference clip has one cow, so Re-ID needs crops from other videos (or the milking-parlor group) to tell cows apart.
- The cow's AnimalId is unknown; she is labelled locally (`Pen2_20260926_Cow01`) until it can be linked.
