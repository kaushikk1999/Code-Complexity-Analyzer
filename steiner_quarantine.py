"""
Steiner Quarantine Protocol — optimized min-cut solution.

Core idea: "minimum walls to isolate a chosen set of Sectors from every
Hospital" is a minimum vertex cut. Model each traversable cell as an
in->out edge (capacity 1 for '.', INF for 'S'/'H'), adjacency as INF edges,
SOURCE -> chosen Sectors, every Hospital -> SINK. Because Sectors <= 15,
enumerate subsets (largest first) and keep the biggest one whose min cut
<= max_walls.

Speed tricks vs a naive Dinic:
  * explicit-stack DFS augmenting paths (no recursion, no recursion-limit risk)
  * build the graph ONCE, reset capacities with a slice copy per subset
  * cap the flow at max_walls+1 -> each check stops after <=11 augmentations
  * scan subset sizes high->low and return on the first feasible size
"""
from itertools import combinations


def max_safe_sectors(city_map, max_walls):
    R = len(city_map)
    C = len(city_map[0]) if R else 0
    idx, ch = {}, []
    for r in range(R):
        for c in range(C):
            if city_map[r][c] != '#':
                idx[(r, c)] = len(ch)
                ch.append(city_map[r][c])
    N = len(ch)
    SRC, SNK, V = 2 * N, 2 * N + 1, 2 * N + 2
    INF = 1 << 30
    to, cap, head = [], [], [[] for _ in range(V)]

    def add(u, v, w):
        head[u].append(len(to)); to.append(v); cap.append(w)
        head[v].append(len(to)); to.append(u); cap.append(0)

    s_edge = []
    for (r, c), i in idx.items():
        add(2 * i, 2 * i + 1, 1 if ch[i] == '.' else INF)      # vertex capacity
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            j = idx.get((r + dr, c + dc))
            if j is not None:
                add(2 * i + 1, 2 * j, INF)                     # adjacency (INF)
        if ch[i] == 'S':
            s_edge.append(len(to)); add(SRC, 2 * i, 0)         # toggled per subset
        elif ch[i] == 'H':
            add(2 * i + 1, SNK, INF)                           # hospital -> sink
    cap0 = cap[:]
    S = len(s_edge)

    def feasible():                                            # min cut <= max_walls?
        flow = 0
        while True:
            prev = [-1] * V
            seen = bytearray(V); seen[SRC] = 1
            stack = [SRC]; hit = False
            while stack:                                       # <-- explicit stack DFS
                v = stack.pop()
                if v == SNK:
                    hit = True; break
                for e in head[v]:
                    w = to[e]
                    if cap[e] and not seen[w]:
                        seen[w] = 1; prev[w] = e; stack.append(w)
            if not hit:
                return True                                    # residual exhausted
            f, v = INF, SNK
            while v != SRC:
                e = prev[v]; f = min(f, cap[e]); v = to[e ^ 1]
            v = SNK
            while v != SRC:
                e = prev[v]; cap[e] -= f; cap[e ^ 1] += f; v = to[e ^ 1]
            flow += f
            if flow > max_walls:
                return False                                   # over budget, stop early

    for size in range(S, -1, -1):
        for combo in combinations(range(S), size):
            cap[:] = cap0
            for i in combo:
                cap[s_edge[i]] = INF
            if feasible():
                return size
    return 0


if __name__ == "__main__":
    print(max_safe_sectors(["S.H", "#.#", "S.H"], 2))  # -> 2
