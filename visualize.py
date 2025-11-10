# visualize.py
import math
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np

import torch
from arc_dataset import make_eval_dataset, make_loader, K_MAX
from models import UNetARC

ARC_PALETTE = np.array([
    [0, 0, 0], [0, 0, 255], [0, 255, 0], [255, 0, 0], [255, 255, 0],
    [255, 165, 0], [255, 192, 203], [128, 0, 128], [128, 128, 128], [0, 255, 255]
], dtype=np.uint8)

def idgrid_to_rgb(grid_hw: torch.Tensor) -> np.ndarray:
    g = grid_hw.detach().cpu().clone()
    mask = (g < 0)
    g[mask] = 8
    rgb = ARC_PALETTE[g.clamp(0, 9).numpy()]
    if mask.any():
        rgb[mask.numpy()] = np.array([200, 200, 200], dtype=np.uint8)
    return rgb

@torch.no_grad()
def viz_row_batch(model, loader, device="cuda", num_rows=8, out_path="viz_row.png"):
    """
    Shows one horizontal row per sample:
    [Test Input | Prediction | Ground Truth]
    """
    model.eval()
    batch = next(iter(loader))
    X = batch["test_input"].to(device).float()  # [B,10,Ht,30]
    logits = model(X)                           # [B,10,30,30]
    preds = logits.argmax(1)                    # [B,30,30]
    Y = batch.get("target", None)
    if Y is not None:
        Y = Y.to(device).long()

    # Extract the test input (top 30x30 of tall input)
    test_in = X[:, :, 0:30, :].argmax(1).to(torch.long)  # [B,30,30]

    B = min(num_rows, X.shape[0])
    fig, axes = plt.subplots(B, 3, figsize=(9, 3*B))
    if B == 1:
        axes = np.expand_dims(axes, 0)

    for i in range(B):
        inp_rgb = idgrid_to_rgb(test_in[i])
        pred_rgb = idgrid_to_rgb(preds[i])
        gt_rgb = idgrid_to_rgb(Y[i]) if Y is not None else None

        axes[i,0].imshow(inp_rgb);  axes[i,0].set_title(f"Input ({batch['id'][i]})"); axes[i,0].axis("off")
        axes[i,1].imshow(pred_rgb); axes[i,1].set_title("Prediction"); axes[i,1].axis("off")
        if gt_rgb is not None:
            axes[i,2].imshow(gt_rgb); axes[i,2].set_title("Ground Truth"); axes[i,2].axis("off")
        else:
            axes[i,2].axis("off")

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"✅ Saved {out_path}")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    root = "./arc-prize-2025"
    eval_ds = make_eval_dataset(root)
    eval_loader = make_loader(eval_ds, batch_size=8, shuffle=False)

    model = UNetARC(in_ch=10, base=64, num_classes=10, block_h=30,
                    target_block="top", total_blocks=(1+2*K_MAX)).to(device)

    # Optionally load checkpoint:
    ckpt = torch.load("checkpoints/unetarc_epoch032.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])

    viz_row_batch(model, eval_loader, device=device, num_rows=8, out_path="viz/viz_row.png")