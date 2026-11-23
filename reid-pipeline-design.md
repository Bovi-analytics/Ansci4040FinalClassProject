# Cow Re-ID Pipeline — Design (v2, 2026-11-23)

Based on a review of Sonam525/Re-id-for-tracking (MIT code; calf re-ID under a tracker, >99% of labelled time correct over a week on 4 look-alike calves), adapted to our goal: name cows in calving-pen and parlour-entry video with BoviSync animalIds.

## Core principle (adopted from the repo)
**The tracker follows, it never names. A separate offline identity layer names every track segment.**
- Fragmented tracks are acceptable: every fragment gets named independently, so 250 tracklets for one cow is no longer fatal.
- Appearance must never steer the tracker (in the repo, letting appearance re-assign tracks left 43.8% of frames wrongly labelled).
- When evidence is weak, output "?" or a review flag instead of a guess.

## Stages

| # | Stage | What we do | Source |
|---|---|---|---|
| 1 | Decode | AVI → 5 fps frames (stride = round(30/5)). Time = frame_idx / native_fps, never container timestamps (our AVIs have corrupt tails). Drop frames past the last sane timestamp. | Tol_00e + our finding |
| 2 | Detect | YOLO (COCO "cow") baseline → fine-tune on ~300 labelled frames from our pens, with a separate **calf** class and a **person** class. | ours; repo used fine-tuned RF-DETR-Seg |
| 3 | Track + mask | Baseline: YOLO + ByteTrack (already run: BaselineByteTrack). Masks: SAM 2.1 seeded by detector boxes. Output = anonymous segments (seg_id, frame, mask/box). | repo used SAM 3 + distilled backbone (SAM License, gated) |
| 4 | Canonical crops | Every **15 s** per segment: masked crop, background filled flat (124,116,104), rotated so the body's long axis is vertical (mask PCA), 2–98% contrast stretch on foreground, 224×224. Gate: min mask area, neighbour overlap IoS < 0.10, not over-extended (>1.5× rolling median area). | `canonical_crop` in Tol_32 — reuse as is |
| 5 | Embed | DINOv3 ViT-S/16 (`facebook/dinov3-vits16-pretrain-lvd1689m`), `pooler_output`, L2-normalised. 8 bands per crop: whole + top/middle/bottom thirds × original + 180° rotation → [8, 384] fp16, cached as npz. | Tol_28 — reuse as is. ViT-S matched ViT-7B in the repo. |
| 6 | Match | Score = mean of the top-5 cosine similarities to **individual exemplars** (not averages), on the 3 body thirds, max over the two orientations. | `pair_sim` in Tol_30 |
| 7 | Identity layer | Every 3 min, score each live segment against each candidate cow; assign jointly (Hungarian) so two concurrent segments never share a name; change a label only when accumulated evidence beats the current one by margin 0.015, and then for the whole run (backward + forward); "?" or review flag otherwise. | Tol_30 (simplify; drop calf-specific closed-pen rules) |
| 8 | Pools | Score = 0.6 × **frozen anchor pool** (verified crops) + 0.4 × **temporal pool** (confidently labelled crops, lagged 30 min). Anchors stop the self-fed pool locking in its first error. | Tol_50 + Tol_30 |
| 9 | Non-cow filter | Hide segments whose crops are clearly coloured (staff in overalls); inert at night/IR. Also drop calf-class segments. | Tol_54 |
| 10 | Output | CSV per segment: video, seg_id, start, end, animalId or "?", score, margin, flags. Overlay video for review. | Tol_17 |

## Changes needed for our setting

1. **Candidate set from BoviSync (our version of their "closed pen").** The repo's strongest constraint was that the pen held exactly 4 known calves. For us, BoviSync pen-move records give the cows in a calving pen at a given time. Match only against those, not the whole herd.
2. **Anchors from our own labelled footage.** The `CowN` suffix in file names is NOT a cow identity: the pen videos hold several cows, and `CowN` appears to be the part number of a day's export (CalvingPen3Sep26Cow1.avi has the same size as "Copy of 2026-09-26 00-00-00 Calving Pen 3.avi"). Anchors must come from human-labelled segments. Balance anchors by phase (day / dawn / dusk / night) as Tol_50 does.
3. **Night/IR.** Anchors must include night crops; colour-free matching comes for free from the contrast-stretched crops.
4. **Posture.** Calving cows lie down for long periods. Tag crops lying/standing (box aspect or mask shape) and keep both in the anchor pool. The PCA rotation already normalises orientation.
5. **Cross-camera (calving pen → parlour entry).** The repo is single-camera. Parlour-entry views will differ. Build a small labelled parlour set early and measure cross-camera accuracy before relying on it. Parlour RFID/sort-gate reads, if timestamps line up, can label parlour video automatically.
6. **Calf filter.** Newborn calves must not enter the cow pools.
7. **Open set.** Cows that enter the pen after the gallery was built are not in it. The layer must return "?"/"new" for them, not force a match.

## Validation protocol v1: Calving Pen 3, 26 Sep (Part 1 → Part 2)

**Rule: the gallery may only contain cows (and crops) seen in calving-pen footage from the same day earlier, or from previous days. Nothing from the test video, nothing from later.**

Earliest footage on Drive is 26 Sep, so there are no previous days. Validation is therefore within the day:

| Role | Video | Notes |
|---|---|---|
| Gallery (anchors) | CalvingPen3Sep26Cow1.avi (= Part 1, from 00:00) | Human labels each tracked segment with a cow identity |
| Test (query) | CalvingPen3Sep26Cow2.avi (= Part 2, follows Part 1; 403,512 clean frames, 3 h 44 min) | Never used for anchors |
| Optional extra gallery | Pen 1 / Pen 2 footage of 26 Sep earlier than the test start | Only if cows moved between pens |

To confirm: Part 2 starts where Part 1 ends (check last frame of Part 1 vs first frame of Part 2).

Steps:
1. Run stages 1–5 on both parts.
2. **Label Part 1** (gallery): per segment, from contact sheets of its crops, assign Cow A, Cow B, … Map to BoviSync animalIds where possible (ear tags, BoviSync pen records for Pen 3 on 26 Sep). Anonymous letters are enough for re-ID validation; the BoviSync mapping can come later.
3. **Label Part 2** (ground truth only, never fed to the system): same procedure, independently. Cows not present in Part 1 are labelled "new".
4. Run the identity layer on Part 2 with: anchor pool = Part 1 labelled crops; temporal pool = Part 2's own confidently labelled crops, lagged 30 min (causal); candidate set = cows labelled in Part 1.
5. Score.

Leakage guards:
- No Part 2 crop in the anchor pool, ever.
- The seam: a cow standing still across the Part 1/Part 2 boundary is trivially matched by position. Report the first 10 min of Part 2 separately from the rest, so appearance-only accuracy is visible.
- Tune thresholds (margin, anchor weight, window) on Part 1 only (e.g. first half vs second half of Part 1), then freeze them before running Part 2.

Metrics (cow-minutes, as in the repo):
- Accuracy of labelled time = correct / (correct + wrong).
- "?" time, silent vs flagged errors, reported separately.
- New-cow rejection: share of "new" cow time labelled "?"/"new" rather than a gallery cow.
- Per phase (night / dawn / day), since both parts are mostly night/IR.

Harder follow-up test: cross-day re-ID (e.g. Pen 1/2/3 11 Nov → 12 Nov) for cows that stay overnight, which is where the repo's "names come back after the night" result applies.

## What NOT to copy
- SAM 3 + FarmSAM student + calf RF-DETR weights: calf-specific, overhead camera, SAM License / gated. Start with YOLO + ByteTrack / SAM 2.1.
- The four closed-pen pigeonhole rules: replaced by the BoviSync / gallery candidate set.
- Databricks-specific paths and widgets: port the logic, not the notebooks.

## Build order
1. Decode + detect + track on Pen 3 26 Sep Part 1 and Part 2.
2. Canonical crops + 8-band embeddings (port Tol_32 `canonical_crop` and Tol_28).
3. Contact sheets per segment → human labelling of Part 1 (gallery) and Part 2 (ground truth).
4. Matcher + simplified identity layer; run validation protocol v1.
5. BoviSync candidate restriction and animalId mapping.
6. Cross-day test, then parlour-entry pilot.

Licences: repo code MIT (keep the notice when porting). DINOv3 weights under the DINOv3 License (download yourself, gated on HF).
