"""Where the videos and the pipeline outputs live on the group's Shared Drive
(as mounted in Colab). Change it here and every notebook follows.

Note the space at the end of "Calving Pen Videos ".
"""

from pathlib import Path

GroupDir = Path("/content/drive/Shareddrives/BeastsAndBytes/ANSC 4040 Group Project")
RawVideoDir = GroupDir / "Calving Pen Videos "
OutputRoot = GroupDir / "CalvingPenPipelineOutputs"   # one subfolder per video, then one per step
