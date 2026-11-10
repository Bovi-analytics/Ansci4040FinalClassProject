# CalvingPen

Pipeline that identifies individual cows in calving-pen videos. It turns a
long recording into short, clean pictures of each cow and a numeric
"fingerprint" per picture, so the same cow can later be recognised in other
calving videos and in the milking-parlor videos (the other group's part).

It is built one step at a time on one reference recording before it runs on
the rest: `2026-09-26 00-00-00 Calving Pen 2.avi` (4 h 37 min), using the
window **03:35:11 to the end**, which contains one whole calving.

## How it works, in plain words

Each step is one function in `CalvingPipeline/` and one pair of notebooks
("a" = quick test on 4 minutes, "b" = the whole clip), so every step can be
rerun on its own and later on other videos / cows.

```
AVI video ─► 01 cut ─► 02 detect ─► 03 track ─► 04 crop + mask ─► 05 fingerprint ─► 06 pool ─► 07 recognise
             clip      YOLO boxes   SAM masks   cow pictures       384 numbers        all videos  which cow?
```

| Step | Function (file) | What it does | Model / tool | Why |
|---|---|---|---|---|
| 01 Prepare clip | `PrepareClips` (`VideoTools.py`) | Reads the video's details and copies the chosen time window into `ReferenceClip.mp4` | ffmpeg (stream copy, no re-encoding) | The full recording is hours long; we only need the part with the cow |
| 02 Detect | `DetectCows` (`Detect.py`) | In every sampled frame (2 per second), draws a **box** around each cow, with a confidence | **YOLO26n** (Ultralytics, trained on the COCO dataset; only its "cow" class) | Finds where the cows are, frame by frame |
| 03 Track | `TrackCows` (`Track.py`) | Starts from YOLO's boxes and follows each cow's **outline** frame by frame into **tracks**, in 2-minute chunks; resumable | **SAM 2.1 Small, video mode** (Meta's Segment Anything 2.1) | Keeps following her while she lies down or strains, where YOLO loses her |
| 04 Crop + mask | `CropCows` (`Crop.py`) | Drops duplicate and too-short tracks; picks frames spread over each track (≥ 1 per 15 s); cuts the cow out and greys out everything that is not her | **SAM 2.1 Small, image mode** | Clean pictures of only the cow, from many angles |
| 05 Fingerprint | `EmbedCrops` (`Embed.py`) | Turns each picture into 384 numbers describing her coat pattern and shape | **DINOv3 ViT-S/16** (Meta, gated on Hugging Face; open **DINOv2-small** as fallback) | Pictures of the same cow get similar numbers; that is what recognition compares |
| 06 Pool | `BuildPool` (`Pool.py`) | Collects the fingerprints of all videos, labels every track with its cow (`Identities.csv` per video), and checks how well each crop finds its own cow elsewhere | none (reuses step 05) | The reference set to recognise cows against |
| 07 Re-ID | (planned) | For a new picture, finds the most similar pictures in the pool; "unknown" if nothing is close enough | | Answers "which cow is this?" |
| 08 Evaluation | (planned) | Accuracy, unknown-cow rejection, speed | | |
| Baseline | `DetectAndTrack` (`Baselines/ByteTrack.py`) | The first tracker: YOLO boxes linked by position | YOLO26n + **ByteTrack** | Only to show why SAM is used (47 tracks vs. 1) |

None of the models were trained on our cows; all are used as downloaded.

## Glossary

| Term | Meaning |
|---|---|
| **Detection / box** | A rectangle the detector draws around something it thinks is a cow, with a **confidence** from 0 to 1 |
| **Track / TrackId** | One stretch of time in which the tracker followed the same animal from frame to frame. A TrackId is **not a cow**: with SAM (step 03) a new TrackId starts only when a 2-minute chunk cannot be joined to the previous one; with the old ByteTrack baseline a new one started whenever YOLO lost her for more than 10 s. One cow can still have several tracks |
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
| 01 | The AVI header says 150 fps, but the real rate is **30 fps** (the recorder writes ~5 index entries per frame). The copied clip is 1:02:01, 1920×1080, 857 MB. About 93 % of consecutive frames are identical, so the camera updates the picture only ~2 times a second (hence 2 fps from step 02 on) |
| Baseline (ByteTrack) | One cow in the pen, but **47 tracks**: YOLO loses her while she lies down or strains (630–2500 s), and sometimes puts a second, weaker box on her head or rear |
| 03 (SAM 2.1) | On 1080–1980 s, where ByteTrack broke up, SAM kept her as **one track for all 900 s**; YOLO agreed with SAM's box on **97 %** of frames. Test 1080–1320 s at 2 fps: one track for all 240 s, YOLO agrees 86 %, SAM alone 14 %, disagrees 0 % (ByteTrack: 3 tracks, cow in 14 % of frames). ~165 s per 2-minute chunk on a T4, so the hour takes ~75–90 min (half with 1 fps); a stopped run continues where it ended |
| 04 (on the ByteTrack tracks; to redo on SAM) | 167 duplicate boxes removed; **14 of 40 tracks kept**; **306 crops, all the same cow**. SAM cuts her out cleanly. When she licks the calf, the calf can end up inside her mask (track 198) |
| 05 (on the ByteTrack tracks; to redo on SAM) | DINOv3 groups her crops by **pose**: lying crops look alike (0.76–0.86), standing / rear / after-birth crops look alike (0.74–0.84), but lying vs. standing drops to 0.57–0.62 although it is the same cow. So Re-ID must match each crop against her **closest** crops, not an average. Different cows need to score below ~0.55; that can only be checked with a second cow |

ByteTrack track timeline (clip seconds; add 03:35:11 for the time in the video):
T2 3–48 standing · T5 51–401 lying · T27 400–481 lying · T37 475–513 lying · T40 513–630 standing ·
T54 1081–1097 standing, rear · T57 1117–1194 lying · T125 1982–2003 lying · T156 2199–2263 lying ·
T169 2302–2409 lying · T186 2477–2513 standing, just after the birth · T198 2516–3256 licking the calf ·
T234 3265–3623 standing, rear · T253 3627–3714 standing, close to the camera.
The calf appears to be born around 04:15–04:16 video time.

## Where things live

| What | Where |
|---|---|
| Pipeline code (`CalvingPipeline/*.py`) | this repo |
| Notebooks (`Step*.ipynb`, `BaselineByteTrack.ipynb`) | repo root, saved **without outputs**. Open them in Colab from GitHub; save your run with File > Save a copy in Drive, never back to GitHub |
| Videos, clips, previews, CSVs | Google Drive, never committed |

Each notebook clones this repo in its first cell and imports `CalvingPipeline`.

Outputs go to the group's Shared Drive, `ANSC4040GroupProject/CalvingPenPipelineOutputs/`
(set once in `CalvingPipeline/Paths.py`), one folder per video, one subfolder per step.
The folder is shared, so one person runs a given step at a time:

```
<video name>/
├── ReferenceClip.mp4, Probe.json          01
├── Step02aDetectTest/, Step02bDetectFull/ 02: Detections.csv, DetectionSummary.csv, Detect.json, PreviewDetections.mp4
├── Step03aTrackTest/,  Step03bTrackFull/  03: Tracks.csv, TrackSummary.csv, Comparison.csv, PreviewTracks.mp4, Chunks/
├── Step04aCropTest/,   Step04bCropFull/   04: KeptTracks.csv, Crops.csv, Crops/, Sheets/
├── Step05aEmbedTest/,  Step05bEmbedFull/  05: Embeddings.npy, EmbeddingIndex.csv, TrackSimilarity.csv/.png
├── BaselineByteTrack/                     baseline: Tracks.csv, TrackSummary.csv, PreviewTracks.mp4
└── Identities.csv                         06 input: which cow each track is (edit by hand; never overwritten)
PoolTest/, Pool/                           06 (next to the video folders): PoolEmbeddings.npy, PoolIndex.csv,
                                           PoolSummary.csv, SelfCheck.csv/.png
```

Folders from the first version (`Step02aTest`, `Step02bFull`, `Step02cSam2`, `Step02dSam2Full`,
`Step03aTest`, `Step03bFull`, `Step04aTest`, `Step04bFull`) are no longer used: rename
`Step02bFull` to `BaselineByteTrack` and move the rest into an `Old/` folder.

## Notebooks

"a" notebooks run on 1080–1320 s (4 minutes); "b" notebooks run the whole clip. Run "a" first after changing settings. Each "a" reads the previous "a", each "b" the previous "b".

| Step | Notebook | Runtime | Status |
|---|---|---|---|
| 01 | `Step01PrepareClip.ipynb` | CPU, a few min | done |
| 02 | `Step02aDetectTest.ipynb` / `Step02bDetectFull.ipynb` | T4, ~1 min / ~5–10 min | to run |
| 03 | `Step03aTrackTest.ipynb` / `Step03bTrackFull.ipynb` (resumable) | T4, ~5 min / ~75–90 min | to run |
| 04 | `Step04aCropTest.ipynb` / `Step04bCropFull.ipynb` | T4, ~1 min / ~10 min | to run |
| 05 | `Step05aEmbedTest.ipynb` / `Step05bEmbedFull.ipynb` | T4, <1 min / ~2 min | to run |
| 06 | `Step06aPoolTest.ipynb` / `Step06bPoolFull.ipynb` | CPU, seconds | to run (delete `Identities.csv` once first) |
| - | `BaselineByteTrack.ipynb`: the old tracker, for comparison only | T4, ~20 min | done |
| 07 | Re-ID prediction | | |
| 08 | Evaluation | | |

## Conventions

- PascalCase for files, folders, functions and variables.
- Never load a whole video into memory; seek or stream.
- Test on a short piece ("a" notebooks) before full recordings.
- Every step overwrites its own output folder; edit the settings cell and rerun instead of adding cells.
- No videos, weights or tokens in git. The Hugging Face token goes in Colab Secrets as `HF_TOKEN`.

## Known issues

- One cow can still become several tracks (see the glossary); Re-ID, not the tracker, decides which tracks are the same cow.
- SAM 2.1 tracking needs ~1.5 h of T4 time for the hour at 2 fps; it saves each chunk, so the run can be split over several sessions or accounts.
- SAM could drift onto the calf or the floor; check `YoloDisagrees` in `TrackSummary.csv` and the preview.
- A calf touching the cow (e.g. being licked) can end up inside her SAM mask (step 04).
- The fingerprint depends strongly on pose (standing vs. lying); see the step 05 result above.
- The reference clip has one cow, so Re-ID needs crops from other videos (or the milking-parlor group) to tell cows apart.
- The cow's AnimalId is unknown; she is labelled locally (`Pen2_20260926_Cow01`) until it can be linked.
- The raw videos were renamed on Drive (Nov 2026); step 01's `VideoPath` must be set to the new name before preparing a new clip.
