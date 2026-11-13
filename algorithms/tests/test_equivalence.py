"""Correctness checks that do not need the TLC files."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from algorithms.fpgrowth import mine_parallel, mine_serial, transactions_to_frame
from algorithms.kmeans_par import kmeans
from algorithms.kmeans_serial import kmeans_numpy


def test_fpgrowth_matches_mlxtend():
    from mlxtend.frequent_patterns import fpgrowth

    rng = np.random.default_rng(0)
    n, n_items, width = 1500, 12, 5
    transactions = []
    for _ in range(n):
        items = rng.choice(n_items, size=width, replace=False).tolist()
        transactions.append(items)
    frame = transactions_to_frame([[str(i) for i in row] for row in transactions])
    # Remap strings back to the column order used by mlxtend.
    index = {name: i for i, name in enumerate(frame.columns)}
    encoded = [[index[str(i)] for i in row] for row in transactions]
    min_support = 0.08
    ours = mine_serial(encoded, min_support)
    names = list(frame.columns)
    ours_named = {frozenset(names[i] for i in key): value for key, value in ours.items()}
    ref = fpgrowth(frame, min_support=min_support, use_colnames=True)
    ref_map = {frozenset(s): float(v) for s, v in zip(ref["itemsets"], ref["support"])}
    assert set(ours_named) == set(ref_map), (len(ours_named), len(ref_map), len(set(ours_named) ^ set(ref_map)))
    for key, support in ours_named.items():
        assert abs(support - ref_map[key]) < 1e-9
    parallel = mine_parallel(encoded, min_support, workers=2)
    assert set(parallel) == set(ours)
    for key, support in parallel.items():
        assert abs(support - ours[key]) < 1e-12


def test_kmeans_threads_match_numpy_and_sklearn():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(600, 4))
    init = X[:4].copy()
    n_iter = 8
    c_np, y_np = kmeans_numpy(X, init, n_iter)
    c1, y1 = kmeans(X, init, n_iter, n_threads=1, block=64)
    c4, y4 = kmeans(X, init, n_iter, n_threads=2, block=32)
    assert np.array_equal(y1, y_np)
    assert np.array_equal(y4, y1)
    assert np.max(np.abs(c1 - c_np)) < 1e-8
    assert np.max(np.abs(c4 - c1)) < 1e-8
    _, y_fast = kmeans(X, init, n_iter, n_threads=2, block=32, parallel_update=True)
    assert adjusted_rand_score(y1, y_fast) >= 0.98
    model = KMeans(n_clusters=4, init=init, n_init=1, max_iter=n_iter, tol=0.0, algorithm="lloyd", random_state=0)
    model.fit(X)
    ari = adjusted_rand_score(y1, model.labels_)
    centroid_gap = np.max(np.abs(np.sort(c1, axis=0) - np.sort(model.cluster_centers_, axis=0)))
    assert ari == 1.0 or centroid_gap < 1e-4, (ari, centroid_gap)


if __name__ == "__main__":
    test_fpgrowth_matches_mlxtend()
    print("fpgrowth ok")
    test_kmeans_threads_match_numpy_and_sklearn()
    print("kmeans ok")
