from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def save_annual_counts(year_counts, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    year_counts.plot(ax=ax, marker="o", linewidth=1.8)
    ax.set_title("Volume annuel de relevés")
    ax.set_xlabel("Année d'observation")
    ax.set_ylabel("Nombre de relevés")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_loss_curves(history: dict[str, list[float]], path: Path, title: str, x_label: str = "Itération") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, values in history.items():
        ax.plot(values, label=label, linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Perte")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_time_curves(curves: dict[str, tuple[list[float], list[float]]], path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, (xs, ys) in curves.items():
        ax.plot(xs, ys, label=label, linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel("Temps écoulé (s)")
    ax.set_ylabel("Perte validation")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_heatmap(matrix, labels: list[str], path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig_size = max(6, min(14, len(labels) * 0.55))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    image = ax.imshow(matrix, cmap="viridis", vmin=0)
    ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=70, ha="right")
    ax.set_yticklabels(labels)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
