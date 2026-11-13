"""Numba prange Lloyd k-means.

The assignment loop is a parallel for over blocks of rows. Centroid updates
use a per-thread accumulator so threads do not share a write location.
`set_num_threads` selects p. The serial baseline of the same kernel is p = 1.
"""

from __future__ import annotations

import numpy as np

from algorithms.kmeans_serial import kmeans_plusplus

_nb = None


def _numba():
    global _nb
    if _nb is None:
        import numba as nb

        _nb = nb
    return _nb


def _compile():
    nb = _numba()

    @nb.njit(parallel=True, cache=False)
    def assign(X, C, block):
        n, d = X.shape
        k = C.shape[0]
        labels = np.empty(n, np.int32)
        nblocks = (n + block - 1) // block
        for b in nb.prange(nblocks):
            start = b * block
            end = min(n, start + block)
            for i in range(start, end):
                best = 1.0e300
                best_j = 0
                for j in range(k):
                    s = 0.0
                    for t in range(d):
                        diff = X[i, t] - C[j, t]
                        s += diff * diff
                    if s < best:
                        best = s
                        best_j = j
                labels[i] = best_j
        return labels

    @nb.njit(parallel=True, cache=False)
    def update(X, labels, k):
        n, d = X.shape
        nthreads = nb.get_num_threads()
        sums = np.zeros((nthreads, k, d), np.float64)
        cnts = np.zeros((nthreads, k), np.int64)
        for i in nb.prange(n):
            tid = nb.get_thread_id()
            j = labels[i]
            cnts[tid, j] += 1
            for t in range(d):
                sums[tid, j, t] += X[i, t]
        out_sum = np.zeros((k, d), np.float64)
        out_cnt = np.zeros(k, np.int64)
        for tid in range(nthreads):
            for j in range(k):
                out_cnt[j] += cnts[tid, j]
                for t in range(d):
                    out_sum[j, t] += sums[tid, j, t]
        return out_sum, out_cnt

    return assign, update


_FUNCS = None


def _funcs():
    global _FUNCS
    if _FUNCS is None:
        _FUNCS = _compile()
    return _FUNCS


def kmeans(X: np.ndarray, init: np.ndarray, n_iter: int, n_threads: int, block: int = 4096, parallel_update: bool = False):
    """Lloyd iterations.

    `parallel_update=False` reduces centroids in fixed row order so labels
    match the NumPy reference for every thread count. The scaling runs set
    `parallel_update=True` to time the thread-local reduction as well.
    """
    nb = _numba()
    nb.set_num_threads(int(max(1, n_threads)))
    assign, update = _funcs()
    X64 = np.ascontiguousarray(X, dtype=np.float64)
    centers = np.array(init, dtype=np.float64, copy=True, order="C")
    labels = np.zeros(X64.shape[0], dtype=np.int32)
    block = int(max(1, block))
    k = centers.shape[0]
    for _ in range(n_iter):
        labels = assign(X64, centers, block)
        if parallel_update:
            sums, cnts = update(X64, labels, k)
            for j in range(k):
                if cnts[j] > 0:
                    centers[j] = sums[j] / cnts[j]
        else:
            for j in range(k):
                mask = labels == j
                if np.any(mask):
                    centers[j] = X64[mask].mean(axis=0)
    return centers, labels


def init_centers(X: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sample_n = min(len(X), 20_000)
    sample = X[rng.choice(len(X), size=sample_n, replace=False)]
    return kmeans_plusplus(sample.astype(np.float64), k, rng)
