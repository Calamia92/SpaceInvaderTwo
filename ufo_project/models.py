from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, log_loss
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 7


@dataclass
class ClassifierResult:
    accuracy: float
    history: dict[str, list[float]]
    elapsed: float
    model: object
    vectorizer: CountVectorizer
    class_labels: list[str]


def _labels(series, classes: list[str]) -> np.ndarray:
    lookup = {label: index for index, label in enumerate(classes)}
    return np.array([lookup[value] for value in series], dtype=np.int64)


def majority_accuracy(valid_df) -> float:
    majority = valid_df["shape"].value_counts().idxmax()
    return float((valid_df["shape"] == majority).mean())


def train_linear_baseline(split, *, epochs: int = 5, max_features: int = 4000) -> ClassifierResult:
    started = time.perf_counter()
    vectorizer = CountVectorizer(max_features=max_features, ngram_range=(1, 1), min_df=2)
    x_train = vectorizer.fit_transform(split.train["comments"])
    x_valid = vectorizer.transform(split.valid["comments"])
    y_train = _labels(split.train["shape"], split.classes)
    y_valid = _labels(split.valid["shape"], split.classes)

    model = SGDClassifier(loss="log_loss", alpha=1e-4, random_state=SEED)
    labels = np.arange(len(split.classes))
    rng = np.random.default_rng(SEED)
    history = {"train": [], "validation": []}
    for _ in range(epochs):
        order = rng.permutation(len(y_train))
        model.partial_fit(x_train[order], y_train[order], classes=labels)
        history["train"].append(float(log_loss(y_train, model.predict_proba(x_train), labels=labels)))
        history["validation"].append(float(log_loss(y_valid, model.predict_proba(x_valid), labels=labels)))

    accuracy = float(accuracy_score(y_valid, model.predict(x_valid)))
    return ClassifierResult(accuracy, history, time.perf_counter() - started, model, vectorizer, split.classes)


class BowMlp(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 192),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(192, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _dense_counts(vectorizer: CountVectorizer, texts) -> torch.Tensor:
    return torch.tensor(vectorizer.transform(texts).toarray(), dtype=torch.float32)


def predict_torch(result: ClassifierResult, texts) -> np.ndarray:
    model = result.model
    assert isinstance(model, nn.Module)
    model.eval()
    x = _dense_counts(result.vectorizer, texts)
    with torch.no_grad():
        return model(x).argmax(dim=1).numpy()


def train_torch_bow(split, *, epochs: int = 8, batch_size: int = 128, max_features: int = 6000) -> ClassifierResult:
    started = time.perf_counter()
    vectorizer = CountVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=2)
    vectorizer.fit(split.train["comments"])
    x_train = _dense_counts(vectorizer, split.train["comments"])
    x_valid = _dense_counts(vectorizer, split.valid["comments"])
    y_train = torch.tensor(_labels(split.train["shape"], split.classes), dtype=torch.long)
    y_valid = torch.tensor(_labels(split.valid["shape"], split.classes), dtype=torch.long)

    torch.manual_seed(SEED)
    model = BowMlp(x_train.shape[1], len(split.classes))
    optimiser = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    history = {"train": [], "validation": []}

    for _ in range(epochs):
        model.train()
        losses = []
        for xb, yb in loader:
            optimiser.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimiser.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            valid_loss = criterion(model(x_valid), y_valid)
        history["train"].append(float(np.mean(losses)))
        history["validation"].append(float(valid_loss))

    with torch.no_grad():
        predictions = model(x_valid).argmax(dim=1).numpy()
    accuracy = float(accuracy_score(y_valid.numpy(), predictions))
    return ClassifierResult(accuracy, history, time.perf_counter() - started, model, vectorizer, split.classes)


def overfit_eight(rows, *, max_iterations: int = 1500) -> tuple[dict[str, list[float]], list[str], int]:
    classes = sorted(rows["shape"].unique())
    class_to_id = {label: index for index, label in enumerate(classes)}

    vectorizer = CountVectorizer()
    x = torch.tensor(vectorizer.fit_transform(rows["comments"]).toarray(), dtype=torch.float32)
    y = torch.tensor([class_to_id[label] for label in rows["shape"]], dtype=torch.long)

    torch.manual_seed(SEED)
    model = nn.Sequential(
        nn.Linear(x.shape[1], 64),
        nn.Tanh(),
        nn.Linear(64, len(classes)),
    )
    optimiser = torch.optim.AdamW(model.parameters(), lr=0.05)
    criterion = nn.CrossEntropyLoss()
    history = {"train": []}
    iterations = max_iterations

    for step in range(1, max_iterations + 1):
        optimiser.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimiser.step()
        history["train"].append(float(loss.detach()))
        if torch.equal(logits.argmax(dim=1), y):
            iterations = step
            break

    with torch.no_grad():
        pred_ids = model(x).argmax(dim=1).numpy()
    predictions = [classes[index] for index in pred_ids]
    return history, predictions, iterations
