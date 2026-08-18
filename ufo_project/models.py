from __future__ import annotations

import numpy as np
import torch
from sklearn.feature_extraction.text import CountVectorizer
from torch import nn

SEED = 7


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
