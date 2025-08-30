import math
import heapq
import matplotlib.pyplot as plt

# ---------------------------
# Data: Cities, positions, and roads (weights = km, arbitrary)
# ---------------------------
CITIES = {
    "Arbor":  (0.0, 0.0),
    "Brook":  (2.0, 1.0),
    "Cedar":  (4.0, 0.0),
    "Dover":  (1.0, 3.0),
    "Elm":    (3.0, 3.0),
    "Fargo":  (5.0, 2.0),
    "Grove":  (6.2, 4.2),
}

# Undirected weighted edges
ROADS = {
    "Arbor": [("Brook", 2.3), ("Dover", 3.3)],
    "Brook": [("Arbor", 2.3), ("Cedar", 2.3), ("Elm", 2.3)],
    "Cedar": [("Brook", 2.3), ("Fargo", 2.3)],
    "Dover": [("Arbor", 3.3), ("Elm", 2.2)],
    "Elm":   [("Dover", 2.2), ("Brook", 2.3), ("Fargo", 2.4)],
    "Fargo": [("Elm", 2.4), ("Cedar", 2.3), ("Grove", 2.5)],
    "Grove": [("Fargo", 2.5)],
}

# Unweighted neighbors (for BFS) and weighted neighbors (for UCS/A*)
def neighbors_unweighted(city):
    return [n for n, _w in ROADS.get(city, [])]

def neighbors_weighted(city):
    return ROADS.get(city, [])

# Straight-line (Euclidean) distance used as an admissible heuristic
def euclidean(a, b):
    ax, ay = CITIES[a]
    bx, by = CITIES[b]
    return math.hypot(ax - bx, ay - by)


# ---------------------------
# BFS (fewest hops, unweighted)
# ---------------------------
from collections import deque

def bfs(start, goal):
    frontier = deque([start])
    parent = {start: None}
    visited = {start}

    while frontier:
        city = frontier.popleft()
        if city == goal:
            return reconstruct_path(parent, goal), len(path_edges(parent, goal))
        for n in neighbors_unweighted(city):
            if n not in visited:
                visited.add(n)
                parent[n] = city
                frontier.append(n)
    return None, math.inf


# ---------------------------
# UCS (optimal with weighted edges)
# ---------------------------
def ucs(start, goal):
    frontier = [(0.0, start)]
    parent = {start: None}
    best_cost = {start: 0.0}

    while frontier:
        g, city = heapq.heappop(frontier)
        if city == goal:
            return reconstruct_path(parent, goal), g
        for n, w in neighbors_weighted(city):
            new_cost = g + w
            if n not in best_cost or new_cost < best_cost[n]:
                best_cost[n] = new_cost
                parent[n] = city
                heapq.heappush(frontier, (new_cost, n))
    return None, math.inf


# ---------------------------
# A* (g + h) with straight-line heuristic
# ---------------------------
def a_star(start, goal, h_fn=euclidean):
    frontier = [(h_fn(start, goal), 0.0, start)]
    parent = {start: None}
    best_cost = {start: 0.0}

    while frontier:
        f, g, city = heapq.heappop(frontier)
        if city == goal:
            return reconstruct_path(parent, goal), g
        for n, w in neighbors_weighted(city):
            ng = g + w
            if n not in best_cost or ng < best_cost[n]:
                best_cost[n] = ng
                parent[n] = city
                nf = ng + h_fn(n, goal)
                heapq.heappush(frontier, (nf, ng, n))
    return None, math.inf


# ---------------------------
# Helpers
# ---------------------------
def reconstruct_path(parent, goal):
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    return list(reversed(path))

def path_edges(parent, goal):
    """Return list of (u,v) edges on the final path"""
    edges = []
    cur = goal
    while parent.get(cur) is not None:
        edges.append((parent[cur], cur))
        cur = parent[cur]
    return list(reversed(edges))


# ---------------------------
# Visualization (pure matplotlib)
# ---------------------------
def draw_graph(path=None, title="City Graph"):
    """
    Draws the graph with node positions, labels, edge weights,
    and (optionally) highlights a 'path' list of city names.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    # Draw edges
    for u, nbrs in ROADS.items():
        x1, y1 = CITIES[u]
        for v, w in nbrs:
            # To avoid double-drawing undirected edges, draw only if u < v by name
            if u < v:
                x2, y2 = CITIES[v]
                ax.plot([x1, x2], [y1, y2], linewidth=1.5)

                # Midpoint for weight label
                mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                ax.text(mx, my, f"{w:.1f}", fontsize=9, ha="center", va="center")

    # Draw nodes
    for city, (x, y) in CITIES.items():
        ax.scatter([x], [y], s=200)
        ax.text(x, y + 0.15, city, ha="center", va="bottom", fontsize=10)

    # Highlight path
    if path and len(path) > 1:
        xs = [CITIES[c][0] for c in path]
        ys = [CITIES[c][1] for c in path]
        ax.plot(xs, ys, linewidth=3.0)   # highlighted route
        # emphasize endpoints
        ax.scatter([xs[0], xs[-1]], [ys[0], ys[-1]], s=300)

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.3)
    plt.show()


# ---------------------------
# Demo runs
# ---------------------------
if __name__ == "__main__":
    start, goal = "Arbor", "Grove"

    bpath, bcost = bfs(start, goal)
    print("BFS:", bpath, "(hops:", bcost, ")")

    upath, ucost = ucs(start, goal)
    print("UCS:", upath, "(km:", round(ucost, 1), ")")

    apath, acost = a_star(start, goal)
    print("A*:", apath, "(km:", round(acost, 1), ")")

    # Visualize A* path by default (change to bpath / upath to compare)
    draw_graph(apath, title=f"A* from {start} to {goal}")
