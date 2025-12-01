import numpy as np
import random
from torch.utils.data import Dataset
import json
from glob import glob
import os

def pad_grid(grid, max_h=30, max_w=30):
    grid = np.array(grid)
    h, w = grid.shape
    padded = np.zeros((max_h, max_w), dtype=int)
    h_end = min(h, max_h)
    w_end = min(w, max_w)
    padded[:h_end, :w_end] = grid[:h_end, :w_end]
    return padded


def generate_random_grid(h=None, w=None):
    if h is None: h = random.randint(3, 10)
    if w is None: w = random.randint(3, 10)

    # Randomly allow very small grids to train Upscale/Tile effectively
    if random.random() < 0.2:
        h = random.randint(1, 4)
        w = random.randint(1, 4)

    # Mode 1: Sparse Objects (better for Object-Centric learning)
    # Create 2-4 distinct distinct colored rectangles/shapes
    g = np.zeros((h, w), dtype=int)
    num_objs = random.randint(2, 4)
    for _ in range(num_objs):
        color = random.randint(1, 9)
        r, c = random.randint(0, h - 1), random.randint(0, w - 1)
        rh, rw = random.randint(1, 4), random.randint(1, 4)
        g[r:min(r + rh, h), c:min(c + rw, w)] = color
    return g


class ARCTaskDataset(Dataset):
    def __init__(self, data_path, mode='train', augment=True):
        self.tasks = []
        self.augment = augment
        if not os.path.exists(data_path):
            print(f"Data not found at {data_path}. Cloning ARC-AGI...")
            os.system("git clone https://github.com/fchollet/ARC-AGI.git")
            data_path = "ARC-AGI/data/training"

        files = glob(os.path.join(data_path, '*.json'))
        print(f"Found {len(files)} real task files.")

        for f in files:
            with open(f, 'r') as fp:
                data = json.load(fp)
            task = {'train': [], 'test': [], 'file': f}
            for pair in data['train']:
                task['train'].append({
                    'input': pad_grid(pair['input']),
                    'output': pad_grid(pair['output'])
                })
            for pair in data['test']:
                task['test'].append({
                    'input': pad_grid(pair['input']),
                    'output': pad_grid(pair['output'])
                })
            self.tasks.append(task)

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        return self.tasks[idx]

if __name__ == "__main__":
    dataset = ARCTaskDataset("ARC-AGI/data/training")