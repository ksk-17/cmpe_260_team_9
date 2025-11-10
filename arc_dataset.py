from __future__ import annotations
import json, math, random
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

ColorInt = int
Grid = List[List[ColorInt]]

NUM_CLASSES = 10
K_MAX = 3
CANVAS_H, CANVAS_W = 30, 30

def grid_to_tensor(grid: Grid) -> torch.LongTensor:
    return torch.as_tensor(grid, dtype=torch.long)

def pad_to(t: torch.Tensor, target_h: int, target_w: int, pad_value: int = -1) -> torch.Tensor:
    """Pad/crop 2D Long tensor (H,W) to (target_h, target_w). Uses -1 as 'empty' by default."""
    h, w = t.shape[-2], t.shape[-1]
    if h > target_h or w > target_w:
        t = t[:target_h, :target_w]
        h, w = t.shape
    pad_bottom = target_h - h
    pad_right  = target_w - w
    return torch.nn.functional.pad(t, (0, pad_right, 0, pad_bottom), value=pad_value)

def one_hot_encoding(t: torch.Tensor, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """
    t: (..., H, W) Long with values in [0..num_classes-1] or -1 for 'empty'.
    Returns: (..., C, H, W) float32 with zeros where t == -1.
    """
    t = torch.as_tensor(t, dtype=torch.long)
    mask = (t >= 0)
    safe = torch.where(mask, t, torch.zeros_like(t))
    oh = torch.nn.functional.one_hot(safe, num_classes=num_classes)  # (..., H, W, C)
    oh = oh.movedim(-1, -3).contiguous().float()                     # (..., C, H, W)
    if mask.ndim == 2:
        oh *= mask.unsqueeze(0)
    else:
        oh *= mask.unsqueeze(-3)
    return oh

def rotate_grid_tensor(t: torch.Tensor, k: int) -> torch.Tensor:
    k = k % 4
    if k == 0: return t
    return torch.rot90(t, k, dims=(-2, -1))

def flip_grid_tensor(t: torch.Tensor, horizontal: bool = True) -> torch.Tensor:
    return torch.flip(t, dims=(-1,)) if horizontal else torch.flip(t, dims=(-2,))

def permute_colors_tensor(t: torch.Tensor, perm: Dict[int, int]) -> torch.Tensor:
    # Works on Long tensor (H,W) with -1 allowed
    out = t.clone()
    mask = out >= 0
    if mask.any():
        flat = out[mask].view(-1)
        # vectorized permutation
        lut = torch.arange(NUM_CLASSES, dtype=torch.long)
        for k, v in perm.items():
            lut[k] = v
        out[mask] = lut[flat]
    return out

class ArcAugment:
    def __init__(
        self,
        rotate: bool = True,
        hflip: bool = True,
        vflip: bool = False,
        color_permute: bool = False,
        seed: Optional[int] = None,
    ):
        self.rotate = rotate
        self.hflip = hflip
        self.vflip = vflip
        self.color_permute = color_permute
        self.rng = random.Random(seed)

    def _maybe_perm(self) -> Optional[Dict[int, int]]:
        if not self.color_permute:
            return None
        palette = list(range(NUM_CLASSES))
        self.rng.shuffle(palette)
        return {i: palette[i] for i in range(NUM_CLASSES)}

    def _aug_grid(self, t: torch.Tensor, perm: Optional[Dict[int, int]]) -> torch.Tensor:
        # rotations
        if self.rotate:
            k = self.rng.choice([0, 1, 2, 3])
            t = rotate_grid_tensor(t, k)
        # flips
        if self.hflip and self.rng.random() < 0.5:
            t = flip_grid_tensor(t, horizontal=True)
        if self.vflip and self.rng.random() < 0.5:
            t = flip_grid_tensor(t, horizontal=False)
        # color permutation
        if perm is not None:
            t = permute_colors_tensor(t, perm)
        return t

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        perm = self._maybe_perm()
        # demo train pairs
        aug_pairs = []
        for p in sample["train_pairs"]:
            inp = self._aug_grid(p["input"], perm)
            out = self._aug_grid(p["output"], perm)
            aug_pairs.append({"input": inp, "output": out})
        sample["train_pairs"] = aug_pairs
        # test input
        sample["test_input"] = self._aug_grid(sample["test_input"], perm)
        # ground truth (if present)
        if "solution" in sample and sample["solution"] is not None:
            sample["solution"] = self._aug_grid(sample["solution"], perm)
        return sample

def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def load_solutions(path: Optional[Path]) -> Dict[str, Grid]:
    if path is None:
        return {}
    data = load_json(path)
    return {str(k): v for k, v in data.items()}

class ArcChallengesDataset(Dataset):
    def __init__(
        self,
        challenges_json: str | Path,
        split_name: str,
        solutions_json: Optional[str | Path] = None,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ):
        self.split_name = split_name
        self.challenges_path = Path(challenges_json)
        self.challenges = load_json(self.challenges_path)
        self.ids: List[str] = sorted(self.challenges.keys())
        self.transform = transform
        self.solutions = load_solutions(Path(solutions_json) if solutions_json else None)

    def __len__(self) -> int:
        return len(self.ids)

    def _make_pairs(self, pairs: List[Dict[str, Grid]]) -> List[Dict[str, torch.Tensor]]:
        out: List[Dict[str, torch.Tensor]] = []
        for p in pairs:
            inp_t = grid_to_tensor(p["input"])
            out_t = grid_to_tensor(p["output"])
            out.append({"input": inp_t, "output": out_t})
        return out

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        task_id = self.ids[idx]
        item = self.challenges[task_id]

        train_pairs = self._make_pairs(item["train"])
        test_input = grid_to_tensor(item["test"][0]["input"])

        sample: Dict[str, Any] = {
            "id": task_id,
            "train_pairs": train_pairs,   # list of {"input": Long[h,w], "output": Long[h,w]}
            "test_input": test_input,     # Long[h,w]
            "split": self.split_name,
        }

        # Attach ground-truth solution if available for this split
        if task_id in self.solutions:
            sols = self.solutions[task_id]  # Grid or List[Grid]
            if isinstance(sols, list):
                sols = sols[0]
            sol = grid_to_tensor(sols)
            sample["solution"] = sol

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

def make_train_dataset(root_dir: str | Path, transform: Optional[Callable] = None) -> ArcChallengesDataset:
    root = Path(root_dir)
    return ArcChallengesDataset(
        challenges_json=root / "arc-agi_training_challenges.json",
        solutions_json=root / "arc-agi_training_solutions.json",
        split_name="train",
        transform=transform,
    )

def make_eval_dataset(root_dir: str | Path, transform: Optional[Callable] = None) -> ArcChallengesDataset:
    root = Path(root_dir)
    return ArcChallengesDataset(
        challenges_json=root / "arc-agi_evaluation_challenges.json",
        solutions_json=root / "arc-agi_evaluation_solutions.json",
        split_name="eval",
        transform=transform,
    )

def make_test_dataset(root_dir: str | Path, transform: Optional[Callable] = None) -> ArcChallengesDataset:
    root = Path(root_dir)
    return ArcChallengesDataset(
        challenges_json=root / "arc-agi_test_challenges.json",
        solutions_json=None,  # hidden on Kaggle
        split_name="test",
        transform=transform,
    )

# ---------- Packing helpers ----------
def pack_one_item_vertical(item: Dict[str, Any],
                           H: int = CANVAS_H,
                           W: int = CANVAS_W,
                           k_max: int = K_MAX) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Returns:
      X: [10, (1+2*k_max)*H, W]  (channels=10, stacked vertically)
      Y: [H, W] or None
    """
    # Build ordered list of grids: test_in, (xin,yout)*k
    blocks: list[torch.Tensor] = []
    test_in = pad_to(item["test_input"], H, W, pad_value=-1)
    blocks.append(one_hot_encoding(test_in))  # [10,H,W]

    pairs = item["train_pairs"][:k_max]
    for p in pairs:
        xin  = pad_to(p["input"],  H, W, pad_value=-1)
        yout = pad_to(p["output"], H, W, pad_value=-1)
        blocks.append(one_hot_encoding(xin))
        blocks.append(one_hot_encoding(yout))

    # If fewer than k_max, append zero blocks
    while len(blocks) < (1 + 2*k_max):
        blocks.append(torch.zeros(NUM_CLASSES, H, W))

    # Concatenate along Height: result [10, (1+2k)*H, W]
    X = torch.cat(blocks, dim=-2).contiguous().float()

    Y = None
    if "solution" in item:
        target = pad_to(item["solution"], H, W, pad_value=-1)
        Y = target  # [H,W] Long (may contain -1)
    return X, Y

def arc_collate(batch: list[Dict[str, Any]], pad_value: int = -1) -> Dict[str, Any]:
    ids = [b["id"] for b in batch]
    splits = [b["split"] for b in batch]
    has_solution = all("solution" in b for b in batch)

    Xs: list[torch.Tensor] = []
    Ys: list[torch.Tensor] = []
    for item in batch:
        X, Y = pack_one_item_vertical(item, H=CANVAS_H, W=CANVAS_W, k_max=K_MAX)
        Xs.append(X)
        if has_solution and Y is not None:
            Ys.append(Y)

    X_batch = torch.stack(Xs, dim=0)     # [B, 10, (1+2K)*H, W]
    out: Dict[str, Any] = {
        "id": ids,
        "split": splits,
        "test_input": X_batch,           # feed this to model(...)
        "hw": (CANVAS_H, CANVAS_W),
        "kmax": K_MAX,
    }
    if has_solution:
        out["target"] = torch.stack(Ys, dim=0)  # [B, H, W]
    return out

def make_loader(
    dataset: Dataset,
    batch_size: int = 16,
    shuffle: bool = False,
    num_workers: int = 0,
    pad_value: int = -1,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda b: arc_collate(b, pad_value=pad_value),
        pin_memory=True,
    )

if __name__ == "__main__":
    root = "./arc-prize-2025"
    augment = ArcAugment(rotate=True, hflip=True, color_permute=True, seed=42)

    train_ds = make_train_dataset(root, transform=augment)
    eval_ds  = make_eval_dataset(root)
    test_ds  = make_test_dataset(root)

    train_loader = make_loader(train_ds, batch_size=8, shuffle=True)
    eval_loader  = make_loader(eval_ds, batch_size=8, shuffle=False)
    test_loader  = make_loader(test_ds, batch_size=8, shuffle=False)

    for batch in train_loader:
        X = batch['test_input']      # [B, 10*(1+2*K_MAX), H, W]
        print("X:", X.shape, X.dtype)
        print("hw:", batch["hw"], "kmax:", batch["kmax"])
        if "target" in batch:
            Y = batch['target']      # [B, H, W]
            print("Y:", Y.shape, Y.dtype, "has -1:", (Y == -1).any().item())
        break