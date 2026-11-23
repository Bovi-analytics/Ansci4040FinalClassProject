"""Cow re-identification, following reid-pipeline-design.md (v2): the tracker
only follows (anonymous TrackIds), an offline identity layer names every track.

Steps (ReidValidation.ipynb runs them):
1. ``Segments.TrackAndCrop``: frames at 5 fps with times from the AVI chunks, YOLO + ByteTrack,
   SAM 2.1 masks and canonical crops every 15 s (one pass over the video).
2. ``Bands.EmbedBands``: 8-band DINOv3 ViT-S/16 fingerprints per crop.
3. ``Labels``: contact sheets, suggested groups and Labels.csv for naming tracks by eye.
4. ``Identity.NameTracks``: matcher (top-5 exemplars, 3 body thirds) + identity layer
   (3-min windows, joint assignment, hysteresis, "?" / "new").
5. ``Validate``: tune on the gallery only, name the test video, score in cow-minutes.

Parts are ported from Sonam525/Re-id-for-tracking (MIT licence): see NOTICE.md.
"""
