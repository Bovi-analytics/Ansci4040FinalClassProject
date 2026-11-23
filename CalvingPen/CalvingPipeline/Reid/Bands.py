"""8-band fingerprints of every canonical crop (stage 5 of the re-ID design;
ported from Tol_28_build_bands.py of Sonam525/Re-id-for-tracking, MIT licence).

Per crop, DINOv3 ViT-S/16 (``pooler_output``, L2-normalised, 384 numbers) of 8
pictures: the whole crop and its top / middle / bottom thirds, each as is and
turned 180 degrees -> an array [8, 384] (float16). Band order:
0 whole, 1 top, 2 middle, 3 bottom, 4-7 the same of the turned crop.

Output ``Bands.npz`` next to ``Crops.csv``: ``CropPath`` (same order as the
rows it covers) and ``Bands`` [n, 8, 384]. Skipped when it is up to date.
"""

import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from ..Embed import DinoV3, LoadEmbedder


def SliceBands(Image_):
    A = np.array(Image_)
    H = A.shape[0]
    Parts = [A, A[:H // 3], A[H // 3:2 * H // 3], A[2 * H // 3:]]
    return [Image.fromarray(P).resize((224, 224), Image.BILINEAR) for P in Parts]


def _CropsMd5(VideoDir) -> str:
    return hashlib.md5((Path(VideoDir) / "Crops.csv").read_bytes()).hexdigest()


def LoadBands(VideoDir):
    """``(Crops table, Bands [n, 8, 384] float32, per band L2-normalised)``,
    rows aligned."""
    VideoDir = Path(VideoDir)
    Crops = pd.read_csv(VideoDir / "Crops.csv")
    Data = np.load(VideoDir / "Bands.npz", allow_pickle=False)
    Index = {P: I for I, P in enumerate(Data["CropPath"].tolist())}
    Keep = Crops.CropPath.map(Index)
    Crops = Crops[Keep.notna()].reset_index(drop=True)
    Bands = Data["Bands"][Keep.dropna().astype(int).to_numpy()].astype(np.float32)
    Bands /= np.linalg.norm(Bands, axis=2, keepdims=True) + 1e-8
    return Crops, Bands


def EmbedBands(VideoDir, Token=None, ModelId=DinoV3, BatchCrops=16, Redo=False) -> dict:
    import torch

    VideoDir = Path(VideoDir)
    OutPath = VideoDir / "Bands.npz"
    Md5 = _CropsMd5(VideoDir)
    if OutPath.exists() and not Redo:
        Old = np.load(OutPath, allow_pickle=False)
        if "CropsMd5" in Old and str(Old["CropsMd5"]) == Md5 and str(Old["ModelId"]) == ModelId:
            print(f"{VideoDir.name}: bands up to date ({len(Old['CropPath'])} crops)")
            return {"Crops": int(len(Old["CropPath"])), "Skipped": True}
    Crops = pd.read_csv(VideoDir / "Crops.csv")
    Processor, Model, Used = LoadEmbedder(ModelId, FallbackModelId=None, Token=Token)
    Device = "cuda" if torch.cuda.is_available() else "cpu"
    DType = torch.float16 if Device == "cuda" else torch.float32
    Model = Model.to(Device, DType).eval()
    Paths, Vectors, Started = [], [], time.time()
    for Start in range(0, len(Crops), BatchCrops):
        Batch, Names = [], []
        for P in Crops.CropPath.iloc[Start:Start + BatchCrops]:
            try:
                Img = Image.open(VideoDir / P).convert("RGB")
            except (OSError, FileNotFoundError):
                continue
            Batch += SliceBands(Img) + SliceBands(Img.rotate(180))
            Names.append(P)
        if not Names:
            continue
        Pixels = Processor(images=Batch, return_tensors="pt")["pixel_values"].to(Device, DType)
        with torch.inference_mode():
            E = torch.nn.functional.normalize(Model(pixel_values=Pixels).pooler_output.float(), dim=-1)
        if not torch.isfinite(E).all():
            raise RuntimeError("DINOv3 gave NaN/inf; rerun with the model in float32")
        Vectors.append(E.cpu().numpy().reshape(len(Names), 8, -1))
        Paths += Names
        if (Start // BatchCrops) % 50 == 0:
            print(f"  {Start + len(Names)}/{len(Crops)} crops, {(time.time() - Started) / 60:.1f} min")
    Bands = np.concatenate(Vectors).astype(np.float16)
    np.savez_compressed(OutPath, CropPath=np.array(Paths), Bands=Bands, CropsMd5=np.array(Md5), ModelId=np.array(Used))
    print(f"{VideoDir.name}: {Bands.shape} in {(time.time() - Started) / 60:.1f} min -> {OutPath.name}")
    return {"Crops": len(Paths), "Skipped": False}
