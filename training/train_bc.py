"""
Behavioral Cloning training script.
Trains the RLBot policy network via supervised cross-entropy loss on expert data.

Usage:
  python training/train_bc.py --dataset models/bc_smartbot.pkl --epochs 50
"""
import os, sys, pickle, argparse
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bots.rl_bot import PolicyNetwork


def train_bc(
    dataset_path: str,
    output_path: str = "models/rl_model_bc.pt",
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    hidden: int = 256,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    print(f"Training BC model on {device}")
    print(f"Dataset: {dataset_path}")
    print(f"Epochs: {epochs}, Batch: {batch_size}, LR: {lr}")

    # Load dataset
    with open(dataset_path, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded {len(data)} samples")

    # Convert to tensors — features may be stored as (1, 26), squeeze first dim
    import numpy as np
    X = torch.tensor(np.array([d['features'] for d in data]).squeeze(1), dtype=torch.float32)
    y = torch.tensor([d['action'] for d in data], dtype=torch.long)

    # Shuffle & split
    perm = torch.randperm(len(X))
    split = int(len(X) * 0.9)
    train_idx = perm[:split]
    val_idx = perm[split:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")

    # Model
    model = PolicyNetwork(input_dim=26, hidden=hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training
    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0

        # Shuffle training data each epoch
        perm = torch.randperm(len(X_train))
        for i in range(0, len(X_train), batch_size):
            idx = perm[i:i + batch_size]
            xb = X_train[idx].to(device)
            yb = y_train[idx].to(device)

            logits = model(xb)
            loss = F.cross_entropy(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(idx)
            correct += (logits.argmax(1) == yb).sum().item()

        train_loss = total_loss / len(X_train)
        train_acc = correct / len(X_train)

        # Validation
        model.eval()
        with torch.no_grad():
            xv = X_val.to(device)
            yv = y_val.to(device)
            val_logits = model(xv)
            val_loss = F.cross_entropy(val_logits, yv).item()
            val_acc = (val_logits.argmax(1) == yv).sum().item() / len(yv)

        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}: train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            torch.save({
                'policy': model.state_dict(),
                'val_acc': val_acc,
                'epoch': epoch,
            }, output_path)

    print(f"\nBest val_acc: {best_val_acc:.3f}")
    print(f"Model saved to {output_path}")

    # Also save state_dict only format for RLBot loading
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(model.state_dict(), output_path.replace(".pt", "_policy.pt"))
    print(f"Policy-only saved to {output_path.replace('.pt', '_policy.pt')}")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="models/bc_smartbot.pkl")
    parser.add_argument("--output", type=str, default="models/rl_model_bc.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=256)
    args = parser.parse_args()

    train_bc(
        dataset_path=args.dataset,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        hidden=args.hidden,
    )