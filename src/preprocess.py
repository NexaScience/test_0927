"""
preprocess.py – Dataset loading & augmentation utilities for all experiments.
Implements two datasets:
1. SyntheticRandomVideoDataset – tiny random-tensor videos used for smoke tests.
2. MABe22Dataset – loader for the real MABe22 mice-behaviour video dataset. The
   loader expects the following directory structure (RGB frames or MP4 files):

    <root>/
        train/
            vid_0001.mp4  (or folder of jpg/png frames)
            vid_0002.mp4
            ...
        val/
            ...
        test/
            ...

If instead of MP4 files each video is stored as a folder of frames, simply place
all frames in a sub-directory (e.g. vid_0001/000001.jpg) and the loader will
pick that up automatically.
"""
from typing import Dict, List, Tuple
from pathlib import Path
import random

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.io import read_video
from PIL import Image

################################################################################
# ----------------------------  video transforms  -----------------------------#
################################################################################

class _PerFrameTransform:
    """Wrapper that applies a torchvision transform to each frame individually."""

    def __init__(self, img_transform):
        self.img_transform = img_transform

    def __call__(self, frames: List[torch.Tensor]) -> torch.Tensor:
        processed: List[torch.Tensor] = []
        for fr in frames:
            # fr: Tensor (H,W,C) uint8 in [0,255]
            if isinstance(fr, torch.Tensor):
                fr_pil = Image.fromarray(fr.numpy())
            else:  # already PIL
                fr_pil = fr
            processed.append(self.img_transform(fr_pil))  # -> Tensor (C,H,W) float
        clip = torch.stack(processed, dim=1)  # (C,T,H,W)
        return clip

################################################################################
# ----------------------------  Dataset classes  ------------------------------#
################################################################################

class SyntheticRandomVideoDataset(Dataset):
    """Tiny synthetic dataset – spits out random videos. Used for CI smoke tests."""

    def __init__(self, num_samples: int = 32, clip_len: int = 8, img_size: int = 112, split="train"):
        super().__init__()
        self.num_samples = num_samples
        self.clip_len = clip_len
        self.img_size = img_size
        self.rng = random.Random(0 if split == "train" else 1)
        tf_train = T.Compose(
            [
                T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
            ]
        )
        tf_val = T.Compose([T.Resize(img_size), T.CenterCrop(img_size), T.ToTensor()])
        self.transform = _PerFrameTransform(tf_train if split == "train" else tf_val)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        clip = [
            torch.randint(0, 256, (self.img_size, self.img_size, 3), dtype=torch.uint8)
            for _ in range(self.clip_len)
        ]
        view1 = self.transform(clip)
        view2 = self.transform(clip)
        frame_dist = torch.tensor(random.randint(0, self.clip_len - 1), dtype=torch.long)
        return {"view1": view1, "view2": view2, "frame_dist": frame_dist}


class MABe22Dataset(Dataset):
    """Dataset loader for the MABe22 mice-triplet video corpus.

    For simplicity and robustness the loader supports *either* MP4 files or
    folders containing individual RGB frames. The temporal augmentation (two
    distinct views + frame distance) is performed on-the-fly.
    """

    def __init__(
        self,
        root: Path,
        split: str,
        clip_len: int = 30,
        fps: int = 15,
        transforms: Dict = None,
    ):
        super().__init__()
        self.root = Path(root).expanduser()
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root folder '{self.root}' not found.")
        self.split = split
        self.clip_len = clip_len
        self.fps = fps
        self.video_paths = self._collect_videos()
        if len(self.video_paths) == 0:
            raise RuntimeError(f"No videos found under {self.root}/{split}")

        # build transform pipeline
        self.transforms = self._build_transforms(train=(split == "train")) if transforms is None else transforms

    # --------------------------------------------------------------------- utils
    def _collect_videos(self) -> List[Path]:
        split_dir = self.root / self.split
        mp4s = list(split_dir.rglob("*.mp4")) + list(split_dir.rglob("*.avi"))
        frame_folders = [p for p in split_dir.iterdir() if p.is_dir()]
        return mp4s + frame_folders

    def _build_transforms(self, train: bool):
        if train:
            img_tf = T.Compose(
                [
                    T.RandomResizedCrop(224, scale=(0.8, 1.0)),
                    T.RandomHorizontalFlip(),
                    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
                    T.RandomGrayscale(p=0.2),
                    T.GaussianBlur(kernel_size=3),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        else:
            img_tf = T.Compose(
                [
                    T.Resize(256),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        return _PerFrameTransform(img_tf)

    # ------------------------------------------------------------------ helpers
    def _read_clip_from_video(self, path: Path, start: int, end: int) -> List[torch.Tensor]:
        # Use torchvision.io for mp4 files
        video, _, _ = read_video(str(path), start_pts=None, end_pts=None, pts_unit="sec")
        # video: (T,H,W,C) uint8
        if end >= video.shape[0]:
            # loop the video if not enough frames
            idxs = list(range(start, video.shape[0])) + [video.shape[0] - 1] * (end - video.shape[0] + 1)
            frames = [video[i] for i in idxs]
        else:
            frames = video[start:end]
        return [fr for fr in frames]

    def _read_clip_from_folder(self, folder: Path, start: int, end: int) -> List[torch.Tensor]:
        frame_files = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png"))
        if len(frame_files) < end:
            frame_files = frame_files + [frame_files[-1]] * (end - len(frame_files))
        selected = frame_files[start:end]
        frames: List[torch.Tensor] = []
        for f in selected:
            img = Image.open(f).convert("RGB")
            frames.append(torch.tensor(img))  # will be converted in transform
        return frames

    # ---------------------------------------------------------------- dataset API
    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        vid_path = self.video_paths[idx]
        # For each video sample TWO temporal crops & compute frame distance
        # ------------------------------------------------------------- video meta
        if vid_path.is_file():
            # Read meta by reading entire video once to get frame count (cheap for mp4 headers)
            video, _, _ = read_video(str(vid_path), pts_unit="sec")
            num_frames = video.shape[0]
        else:
            num_frames = len(list(vid_path.glob("*.jpg"))) + len(list(vid_path.glob("*.png")))
            video = None  # loaded later lazily
        if num_frames < self.clip_len + 1:
            raise RuntimeError(f"Video too short ({num_frames} frames): {vid_path}")

        # sample two start indices
        start1 = random.randint(0, num_frames - self.clip_len)
        start2 = random.randint(0, num_frames - self.clip_len)
        frame_dist = abs(start1 - start2)

        end1 = start1 + self.clip_len
        end2 = start2 + self.clip_len

        # -------------------------------- load clips
        if vid_path.is_file():
            if video is None:
                video, _, _ = read_video(str(vid_path), pts_unit="sec")
            clip_np1 = video[start1:end1]
            clip_np2 = video[start2:end2]
            frames1 = [fr for fr in clip_np1]
            frames2 = [fr for fr in clip_np2]
        else:
            frames1 = self._read_clip_from_folder(vid_path, start1, end1)
            frames2 = self._read_clip_from_folder(vid_path, start2, end2)

        view1 = self.transforms(frames1)  # (C,T,H,W)
        view2 = self.transforms(frames2)
        return {
            "view1": view1,
            "view2": view2,
            "frame_dist": torch.tensor(frame_dist, dtype=torch.long),
        }

################################################################################
# ------------------------------  public API  ----------------------------------
################################################################################

def get_dataset(cfg: Dict, split: str):
    """Factory that returns a dataset according to cfg."""
    name = cfg["name"].lower()
    root = Path(cfg.get("root", "./data")).expanduser()
    params = cfg.get("params", {})

    # Synthetic dataset (used in CI / smoke tests)
    if name == "synthetic":
        return SyntheticRandomVideoDataset(split=split, **params)

    # Real MABe22 dataset
    if name in {"mabe22", "mabe", "mabe22_dataset"}:
        return MABe22Dataset(root=root, split=split, **params)

    raise ValueError(f"Dataset '{name}' not recognised in preprocess.get_dataset().")