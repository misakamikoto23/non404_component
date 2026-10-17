"""Dependency-free assignment helpers used by the contest scorer."""
from __future__ import annotations

from collections import defaultdict


def hungarian_max(weights):
    """Return ``[(row, col)]`` maximizing total weight; only positive pairs are kept.

    ``weights`` is a dense list of rows of non-negative numbers. Rows and
    columns may differ in count. Ties resolve deterministically by index order.
    """
    rows = len(weights)
    cols = len(weights[0]) if rows else 0
    if not rows or not cols:
        return []
    transposed = rows > cols
    if transposed:
        weights = [list(column) for column in zip(*weights)]
        rows, cols = cols, rows
    top = max(max(row) for row in weights)
    # Classic O(n^2 m) potentials algorithm on cost = top - weight (rows <= cols).
    cost = [[top - value for value in row] for row in weights]
    inf = float("inf")
    u = [0.0] * (rows + 1)
    v = [0.0] * (cols + 1)
    owner = [0] * (cols + 1)
    way = [0] * (cols + 1)
    for row in range(1, rows + 1):
        owner[0] = row
        column = 0
        minv = [inf] * (cols + 1)
        used = [False] * (cols + 1)
        while True:
            used[column] = True
            current = owner[column]
            delta = inf
            nxt = 0
            for candidate in range(1, cols + 1):
                if used[candidate]:
                    continue
                reduced = cost[current - 1][candidate - 1] - u[current] - v[candidate]
                if reduced < minv[candidate]:
                    minv[candidate] = reduced
                    way[candidate] = column
                if minv[candidate] < delta:
                    delta = minv[candidate]
                    nxt = candidate
            for candidate in range(cols + 1):
                if used[candidate]:
                    u[owner[candidate]] += delta
                    v[candidate] -= delta
                else:
                    minv[candidate] -= delta
            column = nxt
            if owner[column] == 0:
                break
        while column:
            previous = way[column]
            owner[column] = owner[previous]
            column = previous
    pairs = []
    for column in range(1, cols + 1):
        row = owner[column]
        if row and weights[row - 1][column - 1] > 0:
            pairs.append((column - 1, row - 1) if transposed else (row - 1, column - 1))
    return sorted(pairs)


def sparse_max_matching(left, right, weight):
    """Maximum-weight matching over a sparse positive-weight bipartite graph.

    ``weight`` maps ``(left_id, right_id)`` to a positive number; absent pairs
    cannot match. The graph is split into connected components so each
    Hungarian solve stays small even for pages with hundreds of nets.
    """
    adjacency = defaultdict(set)
    for (a, b), value in weight.items():
        if value > 0:
            adjacency[("L", a)].add(("R", b))
            adjacency[("R", b)].add(("L", a))
    seen, matches = set(), []
    order = {("L", a): i for i, a in enumerate(left)} | {("R", b): i for i, b in enumerate(right)}
    for start in [("L", a) for a in left]:
        if start in seen or start not in adjacency:
            continue
        stack, component = [start], []
        seen.add(start)
        while stack:
            node = stack.pop()
            component.append(node)
            for other in adjacency[node]:
                if other not in seen:
                    seen.add(other)
                    stack.append(other)
        rows = sorted((n[1] for n in component if n[0] == "L"), key=lambda a: order[("L", a)])
        cols = sorted((n[1] for n in component if n[0] == "R"), key=lambda b: order[("R", b)])
        dense = [[weight.get((a, b), 0) for b in cols] for a in rows]
        matches.extend((rows[i], cols[j]) for i, j in hungarian_max(dense))
    return matches


def max_cardinality(left_count, right_count, compatible):
    """Size of a maximum bipartite matching (Kuhn's augmenting paths)."""
    edges = [[j for j in range(right_count) if compatible(i, j)] for i in range(left_count)]
    owner = [-1] * right_count

    def augment(i, visited):
        for j in edges[i]:
            if not visited[j]:
                visited[j] = True
                if owner[j] < 0 or augment(owner[j], visited):
                    owner[j] = i
                    return True
        return False

    return sum(augment(i, [False] * right_count) for i in range(left_count))
