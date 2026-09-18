"""Train the 1D-CNN gait classifier on windowed IMU data.

Evaluates on dogs unseen during training (FR-4), matching the honest
generalization protocol from the literature (docs/PRD.md section 3).
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from ml.common.dataset import build_windowed_dataset
from ml.deep_learning.model import GaitCNN


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=shuffle)


def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        logits = model(X_batch)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(y_batch.numpy())
    return np.array(all_targets), np.array(all_preds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "..", "..", "data", "synthetic"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model-out", default=os.path.join(os.path.dirname(__file__), "gait_cnn.pt"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading and windowing dataset...")
    data = build_windowed_dataset(args.data_dir)
    print(f"Train dogs: {data['train_dogs']}")
    print(f"Test dogs (held out): {data['test_dogs']}")
    print(f"Train windows: {data['X_train'].shape[0]}, Test windows: {data['X_test'].shape[0]}")

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(data["y_train"])
    y_test = encoder.transform(data["y_test"])

    train_loader = make_loader(data["X_train"], y_train, args.batch_size, shuffle=True)
    test_loader = make_loader(data["X_test"], y_test, args.batch_size, shuffle=False)

    model = GaitCNN(n_channels=data["X_train"].shape[2], n_classes=len(encoder.classes_)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        loss = train(model, train_loader, optimizer, criterion, device)
        if epoch % 5 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch}/{args.epochs} - loss: {loss:.4f}")

    y_true, y_pred = evaluate(model, test_loader, device)
    print("\n=== Held-out dog evaluation ===")
    print(classification_report(y_true, y_pred, target_names=encoder.classes_, zero_division=0))
    print("Confusion matrix", list(encoder.classes_))
    print(confusion_matrix(y_true, y_pred))

    torch.save(
        {"model_state": model.state_dict(), "classes": list(encoder.classes_)},
        args.model_out,
    )
    print(f"Saved model to {args.model_out}")


if __name__ == "__main__":
    main()
