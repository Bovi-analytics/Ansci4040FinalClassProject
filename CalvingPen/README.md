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
| 03 Track | `TrackCows` (`Track.py`) | Starts from YOLO's boxes and follows each cow's **outline** frame by frame into **tracks**, in 2-minute chunks that carry the cow over from one to the next; resumable; `Half=True` for half precision (faster) | **SAM 2.1 Small, video mode** (Meta's Segment Anything 2.1) | Keeps following her while she lies down or strains, where YOLO loses her |
| 04 Crop + mask | `CropCows` (`Crop.py`) | Drops duplicate and too-short tracks; picks frames spread over each track (≥ 1 per 15 s); cuts the cow out and greys out everything that is not her | **SAM 2.1 Small, image mode** | Clean pictures of only the cow, from many angles |
| 05 Fingerprint | `EmbedCrops` (`Embed.py`) | Turns each picture into 384 numbers describing her coat pattern and shape | **DINOv3 ViT-S/16** (Meta, gated on Hugging Face; open **DINOv2-small** as fallback) | Pictures of the same cow get similar numbers; that is what recognition compares |
| 06 Pool | `BuildPool` (`Pool.py`) | Collects the fingerprints of all videos, labels every track with its cow (`Identities.csv` per video), and checks how well each crop finds its own cow elsewhere | none (reuses step 05) | The reference set to recognise cows against |
| 07 Re-ID | (planned) | For a new picture, finds the most similar pictures in the pool; "unknown" if nothing is close enough | | Answers "which cow is this?" |
| 08 Evaluation | (planned) | Accuracy, unknown-cow rejection, speed | | |
| All steps | `RunVideo` / `RunVideos` (`Pipeline.py`) | Runs 01–05 for one video, or for a list of videos (other cows) and then 06; same functions and folders as the step notebooks | (the models above) | Process new videos in one go once the settings are right |
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
| 02 (YOLO) | Image size matters: on 1080–1320 s (lying, straining) YOLO at its default size 640 found her in only **15 %** of frames, at **1024** in **86 %**. Whole hour at 640: cow in 47 % of frames, 513 s on a T4. The pipeline uses 1024 |
| Baseline (ByteTrack) | One cow in the pen, but **47 tracks**: YOLO loses her while she lies down or strains (630–2500 s), and sometimes puts a second, weaker box on her head or rear |
| 03 (SAM 2.1) | On 1080–1980 s, where ByteTrack broke up, SAM kept her as **one track for all 900 s**; YOLO agreed with SAM's box on **97 %** of frames. Test 1080–1320 s at 2 fps: one track for all 240 s, YOLO agrees 86 %, SAM alone 14 %, disagrees 0 % (ByteTrack: 3 tracks, cow in 14 % of frames). ~165 s per 2-minute chunk on a T4, so the hour takes ~75–90 min (half with 1 fps); a stopped run continues where it ended Full run: chunk 8 (14:00–16:00) was skipped because YOLO was never sure there and each chunk restarted from YOLO; now each chunk carries the cow over from the previous one |
| 04 (on the ByteTrack tracks; to redo on SAM) | 167 duplicate boxes removed; **14 of 40 tracks kept**; **306 crops, all the same cow**. SAM cuts her out cleanly. When she licks the calf, the calf can end up inside her mask (track 198) |
| 05 (on the ByteTrack tracks; to redo on SAM) | DINOv3 groups her crops by **pose**: lying crops look alike (0.76–0.86), standing / rear / after-birth crops look alike (0.74–0.84), but lying vs. standing drops to 0.57–0.62 although it is the same cow. So Re-ID must match each crop against her **closest** crops, not an average. Different cows need to score below ~0.55; that can only be checked with a second cow |

ByteTrack track timeline (clip seconds; add 03:35:11 for the time in the video):
T2 3–48 standing · T5 51–401 lying · T27 400–481 lying · T37 475–513 lying · T40 513–630 standing ·
T54 1081–1097 standing, rear · T57 1117–1194 lying · T125 1982–2003 lying · T156 2199–2263 lying ·
T169 2302–2409 lying · T186 2477–2513 standing, just after the birth · T198 2516–3256 licking the calf ·
T234 3265–3623 standing, rear · T253 3627–3714 standing, close to the camera.
The calf appears to be born around 04:15–04:16 video time.

## Run the pipeline on a new video (for the team)

Everything runs in **`PipelineSteps.ipynb`** (one cell per step), or `RunPipeline.ipynb` for several videos at once.
[Open PipelineSteps in Colab](https://colab.research.google.com/github/chaseshallow/Ansci4040FinalClassProject/blob/main/PipelineSteps.ipynb).

**What you need (once):**
1. Access to the Shared Drive **BeastsAndBytes** > `ANSC4040GroupProject` (ask the group).
2. A GPU runtime in Colab: **Runtime > Change runtime type > T4 GPU**.
3. A Hugging Face token for DINOv3 in Colab Secrets as `HF_TOKEN` ([steps below](#hugging-face-token-for-dinov3-step-05)).

**What you need per video:**

| Setting (cell 2) | What it is | Where to get it |
|---|---|---|
| `VideoFile` | the raw video's file name, e.g. `2026-09-26-CalvingPen3(1).avi` | cell 2 prints a table of the videos in `CalvingPenVideos ` (`HasOutputs` = someone already ran it). The notebook finds the file by name, also if it was moved inside `ANSC4040GroupProject` |
| `VideoStem` | the output folder name | leave `None` (= the file name without `.avi`). Only the reference video uses its old name `2026-09-26 00-00-00 Calving Pen 2` |
| `ClipStart`, `ClipEnd` | when the calving happens in the video (HH:MM:SS) | the group log, or `Timeline.jpg`: step 01 saves one picture every 15 min of the video before cutting |
| `Test` | `True` = 4 minutes from `TestStart`, `False` = the whole clip | run `True` first, then `False` |

**Order:** cells 1–2, then 3 (cut the clip), 4 (YOLO), 5 (SAM), 6 (crops), 7 (fingerprints), 8 (pool). Check each result
before the next (what to look for is in the notebook's last cell). A 1-hour clip takes ~2 h on a T4, mostly cell 5;
if Colab stops during cell 5, rerun it and it continues.

**Rules:** one person per video (tell the group which one you run); only one person runs the pool (cell 8), after the
videos are finished; nobody edits `CalvingPipeline/*.py` in Colab (changes go through GitHub).

**If something is not found:**

| Message | Fix |
|---|---|
| `No video called '…'` | the file name in `VideoFile` is wrong: copy it from the table in cell 2 |
| `… does not exist. Is Drive mounted, and was the folder renamed?` | a Drive folder was renamed: fix the three lines in `CalvingPipeline/Paths.py` (via GitHub) |
| `Run step 0x first` / missing `Detections.csv`, `Tracks.csv`, `Crops.csv` | run the cells in order, with the same `Test` setting |
| `HF_TOKEN: not set` | add the token to Colab Secrets (below) and turn on notebook access |
| Old code after an update | **Runtime > Disconnect and delete runtime**, then run again from cell 1 |

## Hugging Face token for DINOv3 (step 05)

Step 05 uses **DINOv3**, which Meta shares on Hugging Face only after you accept its license. Each person who runs step 05 (in `Step05*`, `PipelineSteps` or `RunPipeline`) needs their own free token, set up once:

1. **Make a Hugging Face account:** <https://huggingface.co/join> (your Cornell email is fine).
2. **Ask for access to the model:** open <https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m>, log in, fill in the short form and accept the license. Approval can take from minutes to a day; the page then says you have access.
3. **Create a token:** <https://huggingface.co/settings/tokens> > **Create new token** > token type **Read** > give it a name (e.g. `Colab`) > **Create token**, then copy it (it starts with `hf_`). It is shown only once; if lost, create a new one.
4. **Add it to Colab as a secret:** in the notebook, click the **key icon (Secrets)** in the left sidebar > **Add new secret** > Name `HF_TOKEN` (exactly this), Value = the token > switch **Notebook access** on. Colab keeps it for your account, so every notebook can use it.
5. **Check:** run the notebook's first cell; it prints `HF_TOKEN: found`. After step 05, `Model.txt` in the output folder should say `facebook/dinov3-vits16-pretrain-lvd1689m`.

**Never** paste the token into a notebook cell, a chat, or GitHub: anyone with it can use your account. If it leaks, delete it at <https://huggingface.co/settings/tokens> and create a new one.

Without a token (or before access is approved) step 05 falls back to the open **DINOv2-small** and says so. Its fingerprints are **not comparable** with DINOv3's, so step 06 stops with an error if videos were fingerprinted with different models: all videos in one pool must use the same model (rerun step 05 for the odd ones).

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
| 02 | `Step02aDetectTest.ipynb` / `Step02bDetectFull.ipynb` | T4, ~1 min / ~10–15 min | rerun at image size 1024 |
| 03 | `Step03aTrackTest.ipynb` / `Step03bTrackFull.ipynb` (resumable) | T4, ~5 min / ~75–90 min | to run |
| 04 | `Step04aCropTest.ipynb` / `Step04bCropFull.ipynb` | T4, ~1 min / ~10 min | to run |
| 05 | `Step05aEmbedTest.ipynb` / `Step05bEmbedFull.ipynb` | T4, <1 min / ~2 min | to run |
| 06 | `Step06aPoolTest.ipynb` / `Step06bPoolFull.ipynb` | CPU, seconds | to run (delete `Identities.csv` once first) |
| - | `BaselineByteTrack.ipynb`: the old tracker, for comparison only | T4, ~20 min | done |
| 01–06 | `PipelineSteps.ipynb`: **all steps in one notebook**, one cell per step, for any video (same functions and folders as the step notebooks) | T4 | ready |
| 01–06 | `RunPipeline.ipynb`: all steps for a list of videos, then the pool | T4, ~1.5–2 h per hour of video | ready |
| 07 | Re-ID prediction | | |
| 08 | Evaluation | | |

## Conventions

- PascalCase for files, folders, functions and variables.
- Never load a whole video into memory; seek or stream.
- Test on a short piece ("a" notebooks) before full recordings.
- Every step overwrites its own output folder for that video; edit the settings cell and rerun instead of adding cells.
  Only exceptions: step 03 continues a run that **stopped part-way** (same settings, same step 02 boxes), and `Identities.csv` (hand-made labels) is never overwritten.
- No videos, weights or tokens in git. The Hugging Face token goes in Colab Secrets as `HF_TOKEN` (see the token section above).

## Known issues

- One cow can still become several tracks (see the glossary); Re-ID, not the tracker, decides which tracks are the same cow.
- SAM 2.1 tracking needs ~1.5 h of T4 time for the hour at 2 fps; it saves each chunk, so the run can be split over several sessions or accounts.
- SAM could drift onto the calf or the floor; check `YoloDisagrees` in `TrackSummary.csv` and the preview.
- A calf touching the cow (e.g. being licked) can end up inside her SAM mask (step 04).
- The fingerprint depends strongly on pose (standing vs. lying); see the step 05 result above.
- The reference clip has one cow, so Re-ID needs crops from other videos (or the milking-parlor group) to tell cows apart.
- The cow's AnimalId is unknown; she is labelled locally (`Pen2_20260926_Cow01`) until it can be linked.
- The raw videos were renamed on Drive (Nov 2026); step 01's `VideoPath` must be set to the new name before preparing a new clip.
