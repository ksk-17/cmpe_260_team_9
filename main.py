import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from arc_dataset import (  # make sure arc_dataset.py has your modified dataset code
    make_train_dataset,
    make_eval_dataset,
    make_loader,
    K_MAX,
)

from models import UNetARC  # make sure models.py has the updated UNetARC class

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    n = 0
    for batch in tqdm(loader, desc="Training Progress: "):
        X = batch["test_input"].to(device).float()   # [B, 10, tallH, 30]
        Y = batch["target"].to(device).long()        # [B, 30, 30]
        logits = model(X)                            # [B, 10, 30, 30]
        loss = F.cross_entropy(logits, Y.clamp_min(0), ignore_index=-1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(1, n)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    for batch in loader:
        X = batch["test_input"].to(device).float()
        if "target" not in batch:
            continue
        Y = batch["target"].to(device).long()
        logits = model(X)
        loss = F.cross_entropy(logits, Y.clamp_min(0), ignore_index=-1)
        preds = logits.argmax(1)
        mask = Y != -1
        correct = (preds[mask] == Y[mask]).float().mean().item() if mask.any() else 0.0
        total_loss += loss.item()
        total_acc += correct
        n += 1
    return (total_loss / max(1, n), total_acc / max(1, n))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="./arc-prize-2025", help="Path to ARC dataset root")
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--base", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--target_block", type=str, default="top", help="'top' or 'bottom'")
    args = parser.parse_args()

    print(f"Training on device: {args.device}")
    print(f"Dataset root: {args.root}")

    # 1. Load datasets
    train_ds = make_train_dataset(args.root)
    eval_ds  = make_eval_dataset(args.root)
    train_loader = make_loader(train_ds, batch_size=args.bs, shuffle=True)
    eval_loader  = make_loader(eval_ds, batch_size=args.bs, shuffle=False)

    # 2. Build model
    model = UNetARC(
        in_ch=10,
        base=args.base,
        num_classes=10,
        block_h=30,
        target_block=args.target_block,
        total_blocks=(1 + 2 * K_MAX),
    ).to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    # 3. Train
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, args.device)
        val_loss, val_acc = evaluate(model, eval_loader, args.device)
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

        # optional: save checkpoints
        if epoch % 8 == 0 or epoch == args.epochs:
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "args": vars(args),
            }
            save_path = Path("checkpoints")
            save_path.mkdir(exist_ok=True)
            torch.save(ckpt, save_path / f"unetarc_epoch{epoch:03d}.pt")

    print("✅ Training finished!")

if __name__ == "__main__":
    main()