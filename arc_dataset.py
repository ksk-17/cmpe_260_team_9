from pathlib import Path
import json

from typing import Dict, Any, List
import numpy as np

import torch
from torch.utils.data import Dataset

TRAIN_CHALLENGES = "arc-prize-2025/arc-agi_training_challenges.json"
TRAIN_SOLUTIONS = "arc-prize-2025/arc-agi_training_solutions.json"
EVAL_CHALLENGES = "arc-prize-2025/arc-agi_evaluation_challenges.json"
EVAL_SOLUTIONS = "arc-prize-2025/arc-agi_evaluation_solutions.json"
TEST_CHALLENGES = "arc-prize-2025/arc-agi_test_challenges.json"

MAX_DEMONSTRATIONS = 5
MAX_H = 30
MAX_W = 30

def load_arc_task(filepath: str) -> Dict[str, Any]:
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def pad_grid(grid: List[List[int]]) -> np.ndarray:
    reshaped_grid = np.full((MAX_H, MAX_W), -1)
    reshaped_grid[:len(grid), :len(grid[0])] = grid
    return reshaped_grid


from pathlib import Path


# ... (other imports)
# MAX_DEMONSTRATIONS, MAX_H, MAX_W, load_arc_task, pad_grid are assumed to be defined.

def get_input_output_pairs_from_folder(folder_path: str):
    inputs, outputs = [], []
    folder = Path(folder_path)
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    for file_path in folder.glob('*.json'):
        try:
            task_data = load_arc_task(str(file_path))

            task_inputs = []
            n_samples = len(task_data['train'])

            for sample in task_data['train']:
                task_inputs.append(pad_grid(sample['input']))
                task_inputs.append(pad_grid(sample['output']))

            n_defaults = max(0, MAX_DEMONSTRATIONS - n_samples)
            for _ in range(n_defaults):
                task_inputs.append(np.full((MAX_H, MAX_W), -1))
                task_inputs.append(np.full((MAX_H, MAX_W), -1))

            task_inputs.append(pad_grid(task_data['test'][0]['input']))

            inputs.append(task_inputs)

            solution_grid = pad_grid(task_data['test'][0]['output'])
            outputs.append(solution_grid)

        except Exception as e:
            print(f"Warning: Skipping file {file_path} due to error: {e}")
            continue

    return inputs, outputs


def get_input_output_pairs(challenges_path: str, solutions_path: str = None):
    challenges_data = load_arc_task(challenges_path)
    solutions_data = load_arc_task(solutions_path) if solutions_path is not None else None

    inputs, outputs = [], []

    for key, value in challenges_data.items():
        task = []
        n_samples = len(value['train'])
        for sample in value['train']:
            task.append(pad_grid(sample['input']))
            task.append(pad_grid(sample['output']))
        
        n_defaults = max(0, MAX_DEMONSTRATIONS - n_samples)
        for _ in range(n_defaults):
            task.append(np.full((MAX_H, MAX_W), -1))
            task.append(np.full((MAX_H, MAX_W), -1))

        task.append(pad_grid(value['test'][0]['input']))
        
        inputs.append(task)

    if solutions_data is not None:
        for key, value in solutions_data.items():
            outputs.append(pad_grid(value[0]))

    return inputs, outputs

class ARCDataset(Dataset):
    def __init__(self, challenges_path: str = None, solutions_path: str = None, folder_path: str = None):
        if folder_path is None:
            self.inputs, self.outputs = get_input_output_pairs(challenges_path, solutions_path)
        else:
            self.inputs, self.outputs = get_input_output_pairs_from_folder(folder_path)
        self._has_labels = len(self.outputs) == len(self.inputs) and len(self.outputs) > 0

    def __len__(self) -> int:
        return len(self.inputs)
    
    def to_tensor(self, grid) -> torch.tensor:
        return torch.tensor(grid, dtype=torch.long)

    def to_stacked_tensor(self, inputs: List[np.ndarray]) -> torch.Tensor:
        tensor_list = [self.to_tensor(grid) for grid in inputs]
        return torch.stack(tensor_list, dim=0)
    
    def __getitem__(self, idx):
        x = [g for g in self.inputs[idx]]
        x = self.to_stacked_tensor(x)
        if self._has_labels:
            y = self.to_tensor(self.outputs[idx])
            return x, y
        return x, None

if __name__ == "__main__":
    train_dataset = ARCDataset(folder_path='ARC-AGI/data/training')
    test_dataset = ARCDataset(folder_path='ARC-AGI/data/evaluation')
    # train_dataset = ARCDataset(challenges_path=TRAIN_CHALLENGES, solutions_path=TRAIN_SOLUTIONS)
    # test_dataset = ARCDataset(challenges_path=TEST_CHALLENGES)
    for batch in train_dataset:
        print(batch[0].shape, batch[1].shape)
        break
    for batch in test_dataset:
        print(batch[0].shape, batch[1].shape)
        break