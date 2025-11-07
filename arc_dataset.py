# arc_dataset.py
from __future__ import annotations
import json, math, random
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

ColorInt = int  # 0..9
Grid = List[List[ColorInt]]  # rectangular, sizes up to 30x30

def grid_to_tensor(grid: Grid) -> torch.LongTensor:
    return torch.as_tensor(grid, dtype=torch.long)

def pad_to(t: torch.Tensor, target_h: int, target_w: int, pad_value: int = -1) -> torch.Tensor:
    h, w = t.shape[-2], t.shape[-1]
    pad_bottom = target_h - h
    pad_right = target_w - w
    if pad_bottom < 0 or pad_right < 0:
        raise ValueError(f"Target size smaller than tensor: tensor=({h},{w}), target=({target_h},{target_w})")
    # torch.nn.functional.pad pads (left, right, top, bottom)
    return torch.nn.functional.pad(t, (0, pad_right, 0, pad_bottom), value=pad_value)

def rotate_grid_tensor(t: torch.Tensor, k: int) -> torch.Tensor:
    k = k % 4
    if k == 0:
        return t
    if k == 1:
        return torch.rot90(t, 1, dims=(-2, -1))
    if k == 2:
        return torch.rot90(t, 2, dims=(-2, -1))
    if k == 3:
        return torch.rot90(t, 3, dims=(-2, -1))

def flip_grid_tensor(t: torch.Tensor, horizontal: bool = True) -> torch.Tensor:
    return torch.flip(t, dims=(-1,)) if horizontal else torch.flip(t, dims=(-2,))

def permute_colors_tensor(t: torch.Tensor, perm: Dict[int, int]) -> torch.Tensor:
    out = t.clone()
    mask = out >= 0
    out[mask] = out[mask].apply_(lambda v: perm.get(int(v), int(v)))
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
        palette = list(range(10))
        self.rng.shuffle(palette)
        return {i: palette[i] for i in range(10)}

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
    # normalize to python dict[str, grid]
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
            "train_pairs": train_pairs,
            "test_input": test_input,
            "split": self.split_name,
        }

        # Attach ground-truth solution if available for this split
        if task_id in self.solutions:
            sols = self.solutions[task_id]  # could be Grid or List[Grid]
            # normalize to a single grid
            if isinstance(sols, list):   # multiple valid solutions provided
                sols = sols[0]           # choose first deterministically
            sol = grid_to_tensor(sols)
            sample["solution"] = sol

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

def make_train_dataset(
    root_dir: str | Path,
    transform: Optional[Callable] = None
) -> ArcChallengesDataset:
    root = Path(root_dir)
    return ArcChallengesDataset(
        challenges_json=root / "arc-agi_training_challenges.json",
        solutions_json=root / "arc-agi_training_solutions.json",
        split_name="train",
        transform=transform,
    )

def make_eval_dataset(
    root_dir: str | Path,
    transform: Optional[Callable] = None
) -> ArcChallengesDataset:
    root = Path(root_dir)
    return ArcChallengesDataset(
        challenges_json=root / "arc-agi_evaluation_challenges.json",
        solutions_json=root / "arc-agi_evaluation_solutions.json",
        split_name="eval",
        transform=transform,
    )

def make_test_dataset(
    root_dir: str | Path,
    transform: Optional[Callable] = None
) -> ArcChallengesDataset:
    root = Path(root_dir)
    return ArcChallengesDataset(
        challenges_json=root / "arc-agi_test_challenges.json",
        solutions_json=None,  # hidden on Kaggle
        split_name="test",
        transform=transform,
    )

def arc_collate(batch, pad_value: int = -1):
    B = len(batch)
    ids = [b["id"] for b in batch]
    splits = [b["split"] for b in batch]
    has_solution = all("solution" in b for b in batch)

    max_h, max_w, max_T = 1, 1, 1
    for b in batch:
        # demos
        max_T = max(max_T, len(b["train_pairs"]))
        for p in b["train_pairs"]:
            h_i, w_i = p["input"].shape
            h_o, w_o = p["output"].shape
            max_h = max(max_h, h_i, h_o)
            max_w = max(max_w, w_i, w_o)

        # test
        h_t, w_t = b["test_input"].shape
        max_h = max(max_h, h_t); max_w = max(max_w, w_t)

        # solution (robust to accidental leading batch dim)
        if "solution" in b:
            sol = b["solution"]
            if sol.ndim == 3:
                if sol.shape[0] != 1:
                    raise ValueError(
                        f"Each item must carry a single solution grid, but got shape {tuple(sol.shape)}"
                    )
                sol = sol.squeeze(0)
                b["solution"] = sol  # normalize in-place
            h_s, w_s = sol.shape[-2:]
            max_h = max(max_h, h_s); max_w = max(max_w, w_s)

    demo_in = torch.full((B, max_T, max_h, max_w), pad_value, dtype=torch.long)
    demo_out = torch.full((B, max_T, max_h, max_w), pad_value, dtype=torch.long)
    lengths = torch.zeros(B, dtype=torch.long)
    test_batch = torch.full((B, max_h, max_w), pad_value, dtype=torch.long)
    sol_batch = torch.full((B, max_h, max_w), pad_value, dtype=torch.long) if has_solution else None

    for i, b in enumerate(batch):
        T_i = len(b["train_pairs"])
        lengths[i] = T_i
        for t_idx, p in enumerate(b["train_pairs"]):
            demo_in[i, t_idx]  = pad_to(p["input"],  max_h, max_w, pad_value)
            demo_out[i, t_idx] = pad_to(p["output"], max_h, max_w, pad_value)

        test_batch[i] = pad_to(b["test_input"], max_h, max_w, pad_value)

        if has_solution:
            sol = b["solution"]
            if sol.ndim == 3:  # consistent with the check above, but double-sure
                if sol.shape[0] != 1:
                    raise ValueError(f"Expected (H,W) or (1,H,W) per item, got {tuple(sol.shape)}")
                sol = sol.squeeze(0)
            sol_batch[i] = pad_to(sol, max_h, max_w, pad_value)

    out = {
        "id": ids,
        "split": splits,
        "train_pairs": {"input": demo_in, "output": demo_out, "lengths": lengths},
        "test_input": test_batch,
    }
    if has_solution:
        out["solution"] = sol_batch
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

def build_submission_json(
    ids: Iterable[str],
    preds: Iterable[torch.Tensor] | Iterable[np.ndarray] | Iterable[Grid],
) -> Dict[str, Grid]:
    sub: Dict[str, Grid] = {}
    for task_id, g in zip(ids, preds):
        if isinstance(g, torch.Tensor):
            arr = g.detach().cpu().long().numpy()
        elif isinstance(g, np.ndarray):
            arr = g
        else:
            arr = np.asarray(g, dtype=int)
        # ensure ints & nested list
        sub[str(task_id)] = arr.astype(int).tolist()
    return sub


if __name__ == "__main__":
    # 1) Datasets
    root = "./arc-prize-2025"  # folder containing the jsons
    augment = ArcAugment(rotate=True, hflip=True, color_permute=True, seed=42)

    train_ds = make_train_dataset(root, transform=augment)   # has ground-truth "solution"
    eval_ds  = make_eval_dataset(root)                       # has ground-truth "solution"
    test_ds  = make_test_dataset(root)                       # NO solutions

    # 2) Loaders
    train_loader = make_loader(train_ds, batch_size=8, shuffle=True)
    eval_loader  = make_loader(eval_ds, batch_size=8, shuffle=False)
    test_loader  = make_loader(test_ds, batch_size=8, shuffle=False)

    # 3) Inside your training loop (pseudo):
    for batch in train_loader:
        pass

    # 4) Build submission (after inference):
    for batch in test_loader:
        pass
    #     # model predicts `y_hat` of shape (B,H,W) with integer colors in [0..9]
    #     # ... your inference here ...
        
    #     ids.extend(batch["id"])
    #     preds.extend(y_hat.detach().cpu())  # or np arrays

    # submission = build_submission_json(ids, preds)