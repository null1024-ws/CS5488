"""FP-Growth with task-parallel conditional mining.

Item counting is a serial pass over integer transactions. Each frequent item
then grows its own conditional tree. Those trees do not depend on each other,
so they are the parallel tasks. Supports are counts divided by the number of
transactions, matching mlxtend's filter `count / n >= min_support` and the
integer threshold `ceil(min_support * n)` used while pruning a conditional tree.
"""

from __future__ import annotations

import math
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

# Inherited by forked workers. Set only after the tree exists, and only before
# any Numba parallel region in this process.
_TREE = None
_MINSUP = 0


class _Node:
    __slots__ = ("item", "count", "parent", "children")

    def __init__(self, item, parent):
        self.item = item
        self.count = 0
        self.parent = parent
        self.children = {}


class FPTree:
    def __init__(self, rank: dict[int, int]):
        self.root = _Node(None, None)
        self.nodes = defaultdict(list)
        self.cond_items: list[int] = []
        self.rank = rank

    def insert(self, itemset, count=1):
        node = self.root
        node.count += count
        for item in itemset:
            child = node.children.get(item)
            if child is None:
                child = _Node(item, node)
                node.children[item] = child
                self.nodes[item].append(child)
            child.count += count
            node = child

    def is_path(self) -> bool:
        if len(self.root.children) > 1:
            return False
        for nodes in self.nodes.values():
            if len(nodes) > 1 or len(nodes[0].children) > 1:
                return False
        return True

    def conditional(self, item: int, minsup: int) -> "FPTree":
        branches = []
        counts = defaultdict(int)
        for node in self.nodes[item]:
            path = []
            cursor = node.parent
            while cursor is not None and cursor.item is not None:
                path.append(cursor.item)
                cursor = cursor.parent
            path.reverse()
            branches.append((path, node.count))
            for step in path:
                counts[step] += node.count
        items = [it for it, c in counts.items() if c >= minsup]
        items.sort(key=lambda it: counts[it])
        rank = {it: i for i, it in enumerate(items)}
        tree = FPTree(rank)
        for path, count in branches:
            ordered = [it for it in path if it in rank]
            ordered.sort(key=rank.get, reverse=True)
            tree.insert(ordered, count)
        tree.cond_items = self.cond_items + [item]
        return tree


def _mine(tree: FPTree, minsup: int):
    items = list(tree.nodes.keys())
    if tree.is_path():
        for size in range(1, len(items) + 1):
            # Combinations of a path. The support is the minimum node count.
            from itertools import combinations

            for combo in combinations(items, size):
                support = min(tree.nodes[i][0].count for i in combo)
                if support >= minsup:
                    yield support, tree.cond_items + list(combo)
        return
    for item in items:
        support = sum(node.count for node in tree.nodes[item])
        if support >= minsup:
            yield support, tree.cond_items + [item]
    for item in items:
        cond = tree.conditional(item, minsup)
        yield from _mine(cond, minsup)


def _encode(transactions: list[list[int]], min_support: float):
    n = len(transactions)
    if n == 0:
        return {}, 0, n
    counts = defaultdict(int)
    for row in transactions:
        for item in set(row):
            counts[item] += 1
    minsup = math.ceil(min_support * n - 1e-12)
    frequent = {item: c for item, c in counts.items() if c >= minsup}
    # Most frequent first, matching mlxtend's rank sort (ascending rank, reverse insert).
    order = sorted(frequent, key=lambda it: (frequent[it], it))
    rank = {item: i for i, item in enumerate(order)}
    tree = FPTree(rank)
    for row in transactions:
        itemset = [it for it in set(row) if it in rank]
        itemset.sort(key=rank.get, reverse=True)
        if itemset:
            tree.insert(itemset)
    return tree, minsup, n


def _itemsets_from_yields(pairs, n: int, min_support: float) -> dict[frozenset, float]:
    out = {}
    for count, items in pairs:
        support = count / float(n)
        if support + 1e-15 >= min_support:
            out[frozenset(items)] = support
    return out


def mine_serial(transactions: list[list[int]], min_support: float) -> dict[frozenset, float]:
    tree, minsup, n = _encode(transactions, min_support)
    if n == 0 or not isinstance(tree, FPTree):
        return {}
    return _itemsets_from_yields(_mine(tree, minsup), n, min_support)


def _mine_item(item: int):
    cond = _TREE.conditional(item, _MINSUP)
    singles = sum(node.count for node in _TREE.nodes[item])
    found = [(singles, _TREE.cond_items + [item])]
    found.extend(_mine(cond, _MINSUP))
    return found


def mine_parallel(transactions: list[list[int]], min_support: float, workers: int) -> dict[frozenset, float]:
    tree, minsup, n = _encode(transactions, min_support)
    if n == 0 or not isinstance(tree, FPTree):
        return {}
    items = list(tree.nodes.keys())
    if workers <= 1 or len(items) <= 1:
        return _itemsets_from_yields(_mine(tree, minsup), n, min_support)
    global _TREE, _MINSUP
    _TREE = tree
    _MINSUP = minsup
    ctx = __import__("multiprocessing").get_context("fork")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        parts = list(ex.map(_mine_item, items, chunksize=1))
    pairs = [pair for part in parts for pair in part]
    return _itemsets_from_yields(pairs, n, min_support)


def association_rules(itemsets: dict[frozenset, float], min_confidence: float = 0.6, min_lift: float = 1.2):
    from itertools import combinations

    rules = []
    for itemset, support in itemsets.items():
        if len(itemset) < 2:
            continue
        members = list(itemset)
        for r in range(1, len(members)):
            for cons_t in combinations(members, r):
                cons = frozenset(cons_t)
                ant = itemset - cons
                if ant not in itemsets or cons not in itemsets:
                    continue
                conf = support / itemsets[ant]
                lift = conf / itemsets[cons]
                if conf + 1e-15 >= min_confidence and lift > min_lift:
                    rules.append(
                        {
                            "antecedent": ant,
                            "consequent": cons,
                            "support": support,
                            "confidence": conf,
                            "lift": lift,
                        }
                    )
    rules.sort(key=lambda row: (-row["lift"], -row["confidence"], -row["support"]))
    return rules


def transactions_to_frame(transactions: list[list[str]]):
    import pandas as pd

    vocab = sorted({item for row in transactions for item in row})
    index = {item: i for i, item in enumerate(vocab)}
    rows = np_zeros(len(transactions), len(vocab))
    for r, row in enumerate(transactions):
        for item in set(row):
            rows[r, index[item]] = True
    return pd.DataFrame(rows, columns=vocab)


def np_zeros(n, m):
    import numpy as np

    return np.zeros((n, m), dtype=bool)
