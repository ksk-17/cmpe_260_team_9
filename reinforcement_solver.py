import torch
import torch.optim as optim
import numpy as np

from arc_dataset import ARCTaskDataset
from dsl import ARCDSL
from executor_engine import ExecutionEngine
from neural_solver import NeuroSolver
from visualizer import visualize_results
from task_generator import generate_synthetic_batch

def train_agent():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {DEVICE}")

    dsl = ARCDSL()
    engine = ExecutionEngine(dsl)
    model = NeuroSolver(dsl).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("\n--- Phase 1: Synthetic Curriculum Pre-Training ---")
    print("Generating Compositional Tasks (Sequence Length 1-4)...")

    for step in range(2500):  # Increased steps for deeper curriculum
        inputs_np, outputs_np, progs = generate_synthetic_batch(dsl, engine)
        if not inputs_np: continue

        inp_t = torch.tensor(np.array(inputs_np), device=DEVICE)
        out_t = torch.tensor(np.array(outputs_np), device=DEVICE)

        optimizer.zero_grad()
        loss = 0
        for i in range(len(progs)):
            loss += model.train_step_supervised(inp_t[i:i + 1], out_t[i:i + 1], progs[i], DEVICE)

        loss = loss / len(progs)
        loss.backward()
        optimizer.step()

        if step % 200 == 0:
            print(f"Step {step} | Synthetic Loss: {loss.item():.4f}")

    print("\n--- Phase 2: Real ARC Task Training ---")
    dataset = ARCTaskDataset("ARC-AGI/data/training")

    for epoch in range(15):
        total_solved = 0
        solved_indices = []

        for i, task in enumerate(dataset.tasks):
            train_pairs = task['train']
            if not train_pairs: continue

            pair = train_pairs[0]
            inp_t = torch.tensor(pair['input'], device=DEVICE).unsqueeze(0)
            out_t = torch.tensor(pair['output'], device=DEVICE).unsqueeze(0)

            model.eval()
            candidates = []
            with torch.no_grad():
                # Increased Sampling to 50 to find needles in haystack
                for _ in range(50):
                    tokens, _ = model.predict_program(inp_t, out_t, DEVICE)
                    clean_toks = [t for t in tokens if t not in ['<SOS>', '<EOS>', '<PAD>']]
                    candidates.append(clean_toks)
            model.train()

            best_prog = []
            solved = False

            for prog in candidates:
                all_correct = True
                for tp in train_pairs:
                    res = engine.execute(prog, tp['input'])
                    if not np.array_equal(res, tp['output']):
                        all_correct = False
                        break

                if all_correct:
                    solved = True
                    best_prog = prog
                    break

            if solved:
                total_solved += 1
                solved_indices.append(i)
                optimizer.zero_grad()
                loss = model.train_step_supervised(inp_t, out_t, best_prog, DEVICE)
                loss.backward()
                optimizer.step()

            if i % 100 == 0:
                print(f"Ep {epoch} Task {i}: Solved? {solved}")

        print(f"Visualizing results for Epoch {epoch}...")
        visualize_results(model, engine, dataset, DEVICE, num_samples=3, epoch=epoch, solved_indices=solved_indices)
        print(f"Epoch {epoch} Total Solved: {total_solved}/400")

    return model, dsl, engine


if __name__ == "__main__":
    train_agent()