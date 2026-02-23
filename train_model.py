# train_model.py
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# =========================================================
# Model
# =========================================================

class GasCNN(nn.Module):
    """
    Stable CNN:
    Conv -> BatchNorm -> ReLU -> Pool
    prevents exploding/vanishing gradients
    """
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


# =========================================================
# Dataset
# =========================================================

class GasDataset(Dataset):
    """
    Loads dataset.npz produced by data_gen.py
    Applies normalization to stabilize training.
    """
    def __init__(self, path):
        data = np.load(path, allow_pickle=True)
        self.x = data["x"]   # (N,3,H,W)
        self.y = data["y"]

        # class ratio for weighting
        self.pos_ratio = float(np.mean(self.y))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.x[idx]).float()

        # IMPORTANT: normalize gas channel
        gas = x[0]
        gas = gas / (gas.max() + 1e-6)
        x[0] = gas

        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return x, y


# =========================================================
# Metrics
# =========================================================

def compute_metrics(preds, labels):
    tp = ((preds == 1) & (labels == 1)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()

    acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return acc, precision, recall, f1, tp, tn, fp, fn


# =========================================================
# Training
# =========================================================

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = GasDataset(args.data)

    print(f"Dataset size: {len(dataset)}")
    print(f"Unsafe ratio: {dataset.pos_ratio:.3f}")

    n_train = int(0.9 * len(dataset))
    n_val = len(dataset) - n_train

    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    model = GasCNN().to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # -------- class weighting --------
    pos_weight = torch.tensor([(1 - dataset.pos_ratio) / dataset.pos_ratio],
                              device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_acc = 0.0

    # =====================================================
    # Epoch loop
    # =====================================================

    for epoch in range(args.epochs):

        # ------------------ TRAIN ------------------
        model.train()
        running_loss = 0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            x = x.to(device)
            y = y.to(device).unsqueeze(1)

            logits = model(x)
            loss = loss_fn(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            running_loss += loss.item() * x.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        # ------------------ VALID ------------------
        model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)

                logits = model(x)
                probs = torch.sigmoid(logits).squeeze()

                preds = (probs > 0.5).cpu().numpy().astype(int)
                labels = y.numpy().astype(int)

                all_preds.append(preds)
                all_labels.append(labels)

        preds = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)

        acc, precision, recall, f1, tp, tn, fp, fn = compute_metrics(preds, labels)

        print(
            f"\nLoss {train_loss:.4f} | "
            f"Acc {acc:.3f} | "
            f"P {precision:.3f} | "
            f"R {recall:.3f} | "
            f"F1 {f1:.3f}"
        )

        print(f"Confusion: TP={tp} TN={tn} FP={fp} FN={fn}")

        # save best model
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "gas_model_best.pt")
            print("Saved BEST model")

    # final save
    torch.save(model.state_dict(), "gas_model_last.pt")

    print("\nTraining complete.")
    print(f"Best accuracy: {best_acc:.3f}")


# =========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataset/dataset.npz")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=64)

    args = parser.parse_args()
    train(args)