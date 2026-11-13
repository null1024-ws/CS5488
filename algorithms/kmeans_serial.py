"""NumPy Lloyd k-means used as the serial reference."""

from __future__ import annotations

import numpy as np


def kmeans_plusplus(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = X.shape[0]
    centers = np.empty((k, X.shape[1]), dtype=np.float64)
    idx0 = int(rng.integers(0, n))
    centers[0] = X[idx0]
    closest = np.full(n, np.inf)
    for j in range(1, k):
        dist = ((X - centers[j - 1]) ** 2).sum(axis=1)
        closest = np.minimum(closest, dist)
        total = closest.sum()
        if total <= 0:
            centers[j] = X[int(rng.integers(0, n))]
            continue
        probs = closest / total
        centers[j] = X[int(rng.choice(n, p=probs))]
    return centers


def assign_numpy(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    # (n, k) is only used on the small correctness sample.
    d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return d2.argmin(axis=1).astype(np.int32)


def update_numpy(X: np.ndarray, labels: np.ndarray, k: int, centers: np.ndarray) -> np.ndarray:
    out = centers.copy()
    for j in range(k):
        mask = labels == j
        if np.any(mask):
            out[j] = X[mask].mean(axis=0)
    return out


def kmeans_numpy(X: np.ndarray, init: np.ndarray, n_iter: int) -> tuple[np.ndarray, np.ndarray]:
    centers = init.astype(np.float64, copy=True)
    labels = np.zeros(X.shape[0], dtype=np.int32)
    X64 = X.astype(np.float64, copy=False)
    for _ in range(n_iter):
        labels = assign_numpy(X64, centers)
        centers = update_numpy(X64, labels, centers.shape[0], centers)
    return centers, labels
