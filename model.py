# model.py
import argparse
from utils import log_results, set_seed
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os

class SafetyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32x32

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # result: (B,64,16,16)
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # (B,64,1,1)
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.head(z)
        return out

def train_model(dataset_path='dataset.npz', device='cuda', epochs=10, batch_size=32, lr=1e-3):
    dev = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    data = np.load(dataset_path)
    x = data['x']  # (N,2,H,W)
    y = data['y']  # (N,)

    # shuffle & split
    N = len(x)
    idx = np.arange(N)
    np.random.shuffle(idx)
    split = int(0.85 * N)
    train_idx, val_idx = idx[:split], idx[split:]

    model = SafetyCNN().to(dev)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    def batch_iter(indices):
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i+batch_size]
            xb = torch.tensor(x[batch_idx], dtype=torch.float32, device=dev)
            yb = torch.tensor(y[batch_idx], dtype=torch.float32, device=dev).unsqueeze(1)
            yield xb, yb

    for ep in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        for xb, yb in batch_iter(train_idx):
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running_loss += loss.item() * xb.size(0)
        train_loss = running_loss / len(train_idx)

        # validation
        model.eval()
        with torch.no_grad():
            all_logits = []
            all_y = []
            for xb, yb in batch_iter(val_idx):
                logits = model(xb)
                all_logits.append(logits.cpu().numpy())
                all_y.append(yb.cpu().numpy())
            logits = np.vstack(all_logits)
            labels = np.vstack(all_y)
            preds = (1.0 / (1.0 + np.exp(-logits))) > 0.5
            acc = (preds.astype(int) == labels.astype(int)).mean()
        print(f"Epoch {ep}/{epochs}: train_loss={train_loss:.4f}, val_acc={acc:.4f}")

        # checkpoint
        ckpt = f"safety_cnn_ep{ep}.pt"
        torch.save(model.state_dict(), ckpt)
    # final save
    torch.save(model.state_dict(), "safety_cnn.pt")
    print("Training complete. Model saved to safety_cnn.pt")

    return float(acc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset.npz")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    set_seed(args.seed)
    start = time.time()

    acc = train_model(dataset_path=args.dataset,
                      device=args.device,
                      epochs=args.epochs,
                      batch_size=args.batch)

    duration = time.time() - start

    log_results("results.csv", {
        "type": "train",
        "dataset": args.dataset,
        "epochs": args.epochs,
        "acc": acc,
        "time_sec": duration
    })