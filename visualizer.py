import matplotlib.pyplot as plt
import random
import torch
import numpy as np

def visualize_results(model, engine, dataset, device, num_samples=3, epoch=0, solved_indices=None):
    model.eval()

    if solved_indices is None: solved_indices = []
    solved_set = set(solved_indices)
    all_indices = set(range(len(dataset)))
    failed_indices = list(all_indices - solved_set)

    selected_tasks = []

    # 1. Solved
    if solved_indices:
        count = 1 if failed_indices else num_samples
        selected_tasks.extend(random.sample(solved_indices, min(len(solved_indices), count)))

    # 2. Failed
    remaining = num_samples - len(selected_tasks)
    if failed_indices and remaining > 0:
        selected_tasks.extend(random.sample(failed_indices, min(len(failed_indices), remaining)))

    # 3. Fill
    while len(selected_tasks) < num_samples and len(selected_tasks) < len(dataset):
        remaining_opts = list(all_indices - set(selected_tasks))
        if not remaining_opts: break
        selected_tasks.append(random.choice(remaining_opts))

    num_to_plot = len(selected_tasks)
    if num_to_plot == 0: return

    fig, axes = plt.subplots(num_to_plot, 3, figsize=(10, 3 * num_to_plot))
    if num_to_plot == 1:
        axes = np.array([axes])
    elif len(axes.shape) == 1:
        axes = np.expand_dims(axes, 0)

    cmap = plt.cm.colors.ListedColormap([
        '#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
        '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
    ])
    norm = plt.cm.colors.Normalize(vmin=0, vmax=9)

    with torch.no_grad():
        for i, idx in enumerate(selected_tasks):
            task = dataset[idx]
            status = "SOLVED" if idx in solved_set else "FAILED"

            if not task['train']: continue
            train_pair = task['train'][0]
            inp_t = torch.tensor(train_pair['input'], device=device).unsqueeze(0)
            out_t = torch.tensor(train_pair['output'], device=device).unsqueeze(0)

            tokens, _ = model.predict_program(inp_t, out_t, device)
            prog_tokens = [t for t in tokens if t not in ['<SOS>', '<EOS>', '<PAD>']]

            if task['test']:
                test_pair = task['test'][0]
                vis_input = test_pair['input']
                vis_output = test_pair['output']
                set_name = "Test"
            else:
                vis_input = train_pair['input']
                vis_output = train_pair['output']
                set_name = "Train"

            pred_grid = engine.execute(prog_tokens, vis_input)

            def plot_grid(ax, grid, title):
                ax.imshow(grid, cmap=cmap, norm=norm)
                ax.set_title(title, fontsize=10)
                ax.set_xticks([]);
                ax.set_yticks([])
                h, w = grid.shape
                ax.set_xticks(np.arange(-.5, w, 1), minor=True)
                ax.set_yticks(np.arange(-.5, h, 1), minor=True)
                ax.grid(which='minor', color='w', linestyle='-', linewidth=1)

            plot_grid(axes[i, 0], vis_input, f"Task {idx} [{status}]\n{set_name} Input")
            plot_grid(axes[i, 1], pred_grid, "Model Prediction")
            plot_grid(axes[i, 2], vis_output, "Ground Truth")

            prog_str = str(prog_tokens)
            if len(prog_str) > 40: prog_str = prog_str[:37] + "..."
            axes[i, 1].set_xlabel(prog_str, fontsize=8)

    plt.suptitle(f"Epoch {epoch} Visualization", fontsize=14)
    plt.tight_layout()
    plt.show()
    model.train()