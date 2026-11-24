"""The floor of the calving pen in the camera picture (1920 x 1080).

Each camera also sees the next pens: cows behind the front gate, far back in
the barn, and the crowded group pen on the right. In Pen3Sep26 two thirds of
the crops (part 1) and 92 % (part 2) were of those cows, and they made most of
the wrong groups. A crop counts only when the middle of its box's bottom edge
(where the cow stands or lies) is inside the polygon.

``PenAreas``: one polygon per camera view, keyed by the start of the video
name (``CalvingPen3Sep26``). The cameras were moved between September and
November, so a key holds only for the dates it was drawn on. A video without
a polygon is not filtered (a warning is printed). To add or check one,
``DrawPenArea(VideoDir)`` writes ``<VideoDir>/PenArea.jpg``: yellow = the
polygon, green = crops kept, red = crops left out.
"""

from pathlib import Path

import numpy as np
import pandas as pd

PenAreas = {
    "CalvingPen3Sep26": [(0, 705), (90, 690), (590, 645), (1205, 600), (1245, 600), (1140, 1080), (0, 1080)],   # checked on Pen3Sep26 parts 1 and 2
}


def PenAreaFor(VideoFile):
    """Polygon of the camera view ``VideoFile`` starts with (the longest key), or None."""
    Name = Path(str(VideoFile)).name.lower()
    Keys = sorted((K for K in PenAreas if Name.startswith(K.lower())), key=len)
    Area = PenAreas[Keys[-1]] if Keys else None
    if Area is None:
        print(f"No pen area for {Path(str(VideoFile)).name}: cows in the next pens are not left out. "
              "Add a polygon to PenArea.PenAreas (check it with DrawPenArea).")
    return Area


def InPen(Crops, Area) -> pd.Series:
    """True where the middle of the box's bottom edge is inside ``Area``."""
    if Area is None:
        return pd.Series(True, index=Crops.index)
    import cv2
    Polygon = np.array(Area, np.float32)
    return pd.Series([cv2.pointPolygonTest(Polygon, (float((A + B) / 2), float(Y)), False) >= 0
                      for A, B, Y in zip(Crops.X1, Crops.X2, Crops.Y2)], index=Crops.index)


def DrawPenArea(VideoDir, Area=None, FrameName="FirstFrame.jpg", Width=1280) -> Path:
    """``PenArea.jpg``: the polygon and every crop (green kept, red left out)."""
    import cv2
    VideoDir = Path(VideoDir)
    Area = Area if Area is not None else PenAreaFor(__import__("json").loads(
        (VideoDir / "Run.json").read_text())["Settings"]["Video"])
    Image_ = cv2.imread(str(VideoDir / FrameName))
    Crops = pd.read_csv(VideoDir / "Crops.csv")
    Inside = InPen(Crops, Area)
    for (X1, X2, Y2), Ok in zip(Crops[["X1", "X2", "Y2"]].to_numpy(), Inside):
        cv2.circle(Image_, (int((X1 + X2) / 2), int(Y2)), 7, (0, 220, 0) if Ok else (0, 0, 255), -1)
    if Area is not None:
        cv2.polylines(Image_, [np.array(Area, np.int32)], True, (0, 255, 255), 4)
    Scale = Width / Image_.shape[1]
    Out = VideoDir / "PenArea.jpg"
    cv2.imwrite(str(Out), cv2.resize(Image_, None, fx=Scale, fy=Scale, interpolation=cv2.INTER_AREA))
    print(f"{VideoDir.name}: {int(Inside.sum())} of {len(Crops)} crops in the pen -> {Out.name}")
    return Out
