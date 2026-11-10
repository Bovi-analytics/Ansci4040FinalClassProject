"""Step 04: turn every cow crop into a fingerprint vector (an embedding).

DINOv3 ViT-S/16 reads each masked crop (padded to a square with the same
grey as the mask background) and returns 384 numbers that describe its coat
pattern and shape. Vectors are L2-normalized, so the dot product of two of
them is their cosine similarity: near 1 = looks the same, lower = looks
different. Crops of the same cow should be closer to each other than to
crops of other cows; that is what Re-ID (step 06) relies on.

On the reference clip the vectors group by pose more than by track: lying
crops of the cow score ~0.8 with each other but only ~0.6 with her standing
crops. Re-ID should therefore compare against the closest gallery crops,
not a per-cow average.

DINOv3 weights are gated on Hugging Face (request access, then put a read
token in Colab Secrets as HF_TOKEN). Without access, it falls back to the
open DINOv2-small, which also gives 384 numbers per crop.

Outputs in ``OutputDir``, overwritten on every run:

- ``Embeddings.npy``: float32, one row per crop, same order as the index
- ``EmbeddingIndex.csv``: VideoStem, TrackId, Frame, TimeSeconds, CropPath
  (relative to the video's output folder, so videos can be pooled later),
  plus ``NearDuplicate`` (almost identical to the previous crop of its track)
- ``TrackSimilarity.csv`` / ``TrackSimilarity.png``: mean similarity between
  every pair of tracks; one cow = one bright block. With ``GroupSeconds`` set,
  long tracks are split into pieces of that length (``T1@18m`` = track 1,
  from minute 18), so a one-hour SAM track still shows how her look changes
- ``Model.txt``: which model actually produced the vectors
"""

import gc
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

DinoV3 = "facebook/dinov3-vits16-pretrain-lvd1689m"
DinoV2 = "facebook/dinov2-small"
Grey = 127


def LoadEmbedder(ModelId=DinoV3, FallbackModelId=DinoV2, Token=None):
    """``(Processor, Model, UsedModelId)``; falls back if ``ModelId`` cannot be
    loaded (no access yet, no token, too old transformers)."""
    from transformers import AutoImageProcessor, AutoModel

    for Candidate in [ModelId, FallbackModelId]:
        if not Candidate:
            continue
        try:
            Processor = AutoImageProcessor.from_pretrained(Candidate, token=Token)
            Model = AutoModel.from_pretrained(Candidate, token=Token)
            if Candidate != ModelId:
                print(f"Using fallback {Candidate}")
            return Processor, Model, Candidate
        except Exception as Error:  # gated / missing token / unsupported model
            print(f"Could not load {Candidate}: {type(Error).__name__}: {str(Error).splitlines()[0][:200]}")
    raise RuntimeError("No embedding model could be loaded")


def PadToSquare(Image: np.ndarray) -> np.ndarray:
    """Pad with the mask grey so the model's resize never cuts the cow off."""
    H, W = Image.shape[:2]
    Size = max(H, W)
    Square = np.full((Size, Size, 3), Grey, np.uint8)
    Y, X = (Size - H) // 2, (Size - W) // 2
    Square[Y:Y + H, X:X + W] = Image
    return Square


def EmbedImages(Images, Processor, Model, Device, BatchSize=64) -> np.ndarray:
    import torch

    Vectors = []
    for Start in range(0, len(Images), BatchSize):
        Batch = [cv2.cvtColor(PadToSquare(I), cv2.COLOR_BGR2RGB) for I in Images[Start:Start + BatchSize]]
        Inputs = Processor(images=Batch, return_tensors="pt").to(Device)
        with torch.inference_mode():
            Output = Model(**Inputs)
        Vectors.append(Output.pooler_output.float().cpu().numpy())
    Vectors = np.concatenate(Vectors)
    return Vectors / np.linalg.norm(Vectors, axis=1, keepdims=True)


def TrackSimilarity(Vectors: np.ndarray, TrackIds) -> pd.DataFrame:
    """Mean cosine similarity between the crops of every pair of tracks (the
    diagonal leaves out each crop's similarity to itself)."""
    TrackIds = np.asarray(TrackIds)
    Ids = list(dict.fromkeys(TrackIds))
    Similarity = Vectors @ Vectors.T
    Table = np.zeros((len(Ids), len(Ids)))
    for I, A in enumerate(Ids):
        for J, B in enumerate(Ids):
            Block = Similarity[np.ix_(TrackIds == A, TrackIds == B)]
            if A == B:
                N = len(Block)
                Table[I, J] = (Block.sum() - N) / max(N * N - N, 1)
            else:
                Table[I, J] = Block.mean()
    return pd.DataFrame(Table.round(3), index=Ids, columns=Ids)


def _SaveHeatmap(Table: pd.DataFrame, OutPath, Title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Size = 1.2 + 0.45 * len(Table)
    Fig, Ax = plt.subplots(figsize=(max(Size + 1.5, 6), max(Size, 3.5)))   # wide enough for the title
    Image = Ax.imshow(Table.values, cmap="viridis", vmin=min(0.0, Table.values.min()), vmax=1)
    Ax.set_xticks(range(len(Table)), [f"T{I}" for I in Table.columns], rotation=90, fontsize=8)
    Ax.set_yticks(range(len(Table)), [f"T{I}" for I in Table.index], fontsize=8)
    Ax.set_title(Title, fontsize=10)
    Fig.colorbar(Image, ax=Ax, label="mean cosine similarity")
    Fig.tight_layout()
    Fig.savefig(OutPath, dpi=120)
    plt.close(Fig)


def EmbedCrops(CropsDir, OutputDir, VideoStem, ModelId=DinoV3, FallbackModelId=DinoV2, Token=None,
               BatchSize=64, NearDuplicateAbove=0.98, GroupSeconds=None) -> dict:
    """Embed every crop listed in ``CropsDir/Crops.csv`` (a step 03 folder)."""
    import torch

    CropsDir, OutputDir = Path(CropsDir), Path(OutputDir)
    OutputDir.mkdir(parents=True, exist_ok=True)
    Crops = pd.read_csv(CropsDir / "Crops.csv").sort_values(["TrackId", "TimeSeconds"]).reset_index(drop=True)
    Images = [cv2.imread(str(CropsDir / P)) for P in Crops.CropPath]
    Missing = [P for P, I in zip(Crops.CropPath, Images) if I is None]
    if Missing:
        raise FileNotFoundError(f"{len(Missing)} crops missing, e.g. {CropsDir / Missing[0]}")

    Device = "cuda" if torch.cuda.is_available() else "cpu"
    Processor, Model, UsedModel = LoadEmbedder(ModelId, FallbackModelId, Token)
    Model = Model.to(Device).eval()
    Started = time.time()
    Vectors = EmbedImages(Images, Processor, Model, Device, BatchSize)
    Seconds = time.time() - Started
    del Model
    gc.collect()
    if Device == "cuda":
        torch.cuda.empty_cache()

    # Near-duplicates: almost the same picture as the previous crop of the
    # same track (e.g. the cow lying still). Kept, but marked, so later steps
    # can avoid counting the same view many times.
    Previous = np.r_[0, np.einsum("ij,ij->i", Vectors[1:], Vectors[:-1])]
    SameTrack = np.r_[False, Crops.TrackId.values[1:] == Crops.TrackId.values[:-1]]
    Index = pd.DataFrame({
        "VideoStem": VideoStem,
        "TrackId": Crops.TrackId,
        "Frame": Crops.Frame,
        "TimeSeconds": Crops.TimeSeconds,
        "CropPath": [f"{CropsDir.name}/{P}" for P in Crops.CropPath],
        "SimilarityToPrevious": np.where(SameTrack, Previous, np.nan).round(3),
        "NearDuplicate": SameTrack & (Previous > NearDuplicateAbove),
    })

    np.save(OutputDir / "Embeddings.npy", Vectors.astype(np.float32))
    Index.to_csv(OutputDir / "EmbeddingIndex.csv", index=False)
    (OutputDir / "Model.txt").write_text(UsedModel + "\n")
    Groups = Crops.TrackId
    if GroupSeconds:
        Minute = (Crops.TimeSeconds // GroupSeconds * GroupSeconds / 60).astype(int)
        Groups = Crops.TrackId.astype(str) + "@" + Minute.astype(str) + "m"
    Table = TrackSimilarity(Vectors, Groups.to_numpy(dtype=object))
    Table.to_csv(OutputDir / "TrackSimilarity.csv")
    _SaveHeatmap(Table, OutputDir / "TrackSimilarity.png", f"{VideoStem}\n{UsedModel}")

    Off = Table.values[~np.eye(len(Table), dtype=bool)]
    Stats = {"Model": UsedModel, "Crops": len(Index), "Dimensions": int(Vectors.shape[1]),
             "Tracks": len(Table), "NearDuplicates": int(Index.NearDuplicate.sum()),
             "MeanWithinTrack": round(float(np.diag(Table.values).mean()), 3),
             "MeanBetweenTracks": round(float(Off.mean()), 3) if len(Off) else None,
             "CropsPerSecond": round(len(Index) / Seconds, 1)}
    print(Stats)
    return Stats
