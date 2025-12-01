import torch
import torch.optim as optim
from arc_dataset import ARCTaskDataset
from dsl import ARCDSL
from executor_engine import ExecutionEngine
from neural_solver import NeuroSolver
from reinforcement_solver import run_bsil, run_ppo, run_dreamcoder, run_mcts

import matplotlib.pyplot as plt

def run_experiment():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Comparison Experiment on {DEVICE}")

    dataset = ARCTaskDataset("ARC-AGI/data/training")
    dsl = ARCDSL()
    engine = ExecutionEngine(dsl)

    algos = {
        'BSIL': run_bsil,
        'PPO': run_ppo,
        'DreamCoder': run_dreamcoder,
        'MCTS': run_mcts
    }

    results = {name: [] for name in algos}

    for name, train_fn in algos.items():
        print(f"\n--- Testing Algorithm: {name} ---")
        model = NeuroSolver(dsl).to(DEVICE)
        opt = optim.Adam(model.parameters(), lr=1e-3)

        rewards_over_time = []

        for epoch in range(5):
            total_r = 0
            subset_size = min(len(dataset), 50)

            for i in range(subset_size):
                opt.zero_grad()

                # Handle flexible returns
                # PPO/MCTS -> (loss, r)
                # BSIL/DreamCoder -> (loss, r, prog)
                ret = train_fn(model, engine, dataset[i], DEVICE)
                if len(ret) == 3:
                    loss, r, _ = ret
                else:
                    loss, r = ret

                if isinstance(loss, torch.Tensor) and loss.requires_grad:
                    loss.backward()
                    opt.step()
                total_r += r

            avg = total_r / subset_size
            rewards_over_time.append(avg)
            print(f"Ep {epoch}: Avg Reward = {avg:.4f}")

        results[name] = rewards_over_time

    plt.figure(figsize=(10, 6))
    for name, data in results.items():
        plt.plot(data, label=name, marker='o')
    plt.title("RL Algorithm Comparison on ARC Subset")
    plt.xlabel("Epoch")
    plt.ylabel("Avg Reward")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    run_experiment()