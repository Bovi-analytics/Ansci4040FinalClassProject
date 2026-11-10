"""Where the videos and the pipeline outputs live on the group's Shared Drive
(as mounted in Colab). Change it here and every notebook follows.

Note the space at the end of "CalvingPenVideos ".
The group folder was renamed from "ANSC 4040 Group Project" in Nov 2026.
"""

from pathlib import Path

GroupDir = Path("/content/drive/Shareddrives/BeastsAndBytes/ANSC4040GroupProject")
RawVideoDir = GroupDir / "CalvingPenVideos "
OutputRoot = GroupDir / "CalvingPenPipelineOutputs"   # one subfolder per video, then one per step
