"""Reproducible MLP experiments for scikit-learn's digits dataset."""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


DATA_SPLIT_SEED = 42
MODEL_SEED = 42
EPOCHS = 30


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class MLP(nn.Module):
    def __init__(
        self,
        hidden: tuple[int, ...] = (32,),
        activation: str = "sigmoid",
        dropout: float = 0.0,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()
        activations = {
            "sigmoid": nn.Sigmoid,
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "gelu": nn.GELU,
        }
        if activation not in activations:
            raise ValueError(f"Unsupported activation: {activation}")

        layers: list[nn.Module] = []
        in_features = 64
        for width in hidden:
            layers.append(nn.Linear(in_features, width))
            if batch_norm:
                layers.append(nn.BatchNorm1d(width))
            layers.append(activations[activation]())
            if dropout:
                layers.append(nn.Dropout(dropout))
            in_features = width
        layers.append(nn.Linear(in_features, 10))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def load_data() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    features, labels = load_digits(return_X_y=True)
    features = features.astype(np.float32) / 16.0
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=DATA_SPLIT_SEED,
    )
    return (
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(x_test, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
        torch.tensor(y_test, dtype=torch.long),
    )


def train(
    config: dict,
    data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    seed: int = MODEL_SEED,
) -> dict:
    set_seed(seed)
    x_train, x_test, y_train, y_test = data
    model = MLP(
        hidden=config["hidden"],
        activation=config["activation"],
        dropout=config.get("dropout", 0.0),
        batch_norm=config.get("batch_norm", False),
    )
    optimizer_class = torch.optim.Adam if config["optimizer"] == "adam" else torch.optim.SGD
    optimizer = optimizer_class(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config.get("weight_decay", 0.0),
    )
    loss_fn = nn.CrossEntropyLoss()
    history = {"loss": [], "accuracy": []}
    start = time.perf_counter()

    for _ in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(x_train), y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            accuracy = (model(x_test).argmax(dim=1) == y_test).float().mean().item()
        history["loss"].append(loss.item())
        history["accuracy"].append(accuracy)

    history["seconds"] = time.perf_counter() - start
    return history


def save_plot(history: dict, title: str, output_path: Path) -> None:
    epochs = np.arange(1, EPOCHS + 1)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history["loss"], marker="o", markersize=3)
    axes[0].set(title="Training loss", xlabel="Epoch", ylabel="Cross-entropy loss")
    axes[0].grid(alpha=0.3)
    axes[1].plot(epochs, np.array(history["accuracy"]) * 100, color="tab:red", marker="o", markersize=3)
    axes[1].set(title="Test accuracy", xlabel="Epoch", ylabel="Accuracy (%)")
    axes[1].grid(alpha=0.3)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/reproduced"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    baseline = {
        "hidden": (32,),
        "activation": "sigmoid",
        "optimizer": "sgd",
        "lr": 0.1,
        "weight_decay": 0.0,
        "dropout": 0.0,
        "batch_norm": False,
    }
    experiments = [
        ("baseline", baseline),
        ("relu", {**baseline, "activation": "relu"}),
        ("deep_sigmoid", {**baseline, "hidden": (128, 128)}),
        ("adam", {**baseline, "optimizer": "adam", "lr": 0.01}),
        ("small_lr", {**baseline, "lr": 0.01}),
        ("l2", {**baseline, "weight_decay": 1e-4}),
        ("dropout", {**baseline, "dropout": 0.2}),
        ("batchnorm", {**baseline, "batch_norm": True}),
        ("large_lr_failure", {**baseline, "lr": 10.0}),
        (
            "final",
            {
                **baseline,
                "activation": "relu",
                "optimizer": "adam",
                "lr": 0.01,
                "batch_norm": True,
            },
        ),
    ]

    data = load_data()
    rows = []
    for name, config in experiments:
        history = train(config, data)
        save_plot(history, name.replace("_", " ").title(), args.output / f"{name}.png")
        rows.append(
            {
                "experiment": name,
                "test_accuracy_percent": f"{history['accuracy'][-1] * 100:.2f}",
                "training_time_seconds": f"{history['seconds']:.4f}",
                "configuration": str(config),
            }
        )
        print(f"{name:>16}: {history['accuracy'][-1] * 100:6.2f}%")

    with (args.output / "results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
