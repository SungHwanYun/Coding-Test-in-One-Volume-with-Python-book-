def solve(state, sheep, wolf, info, E):
    ret = sheep

    for u in range(len(info)):
        if (state & (1 << u)) == 0:
            continue

        for v in E[u]:
            if state & (1 << v):
               continue

            if info[v] == 0:
                ret = max(ret, solve(state | (1 << v), sheep + 1, wolf, info, E))
            else:
                if sheep > wolf + 1:
                    ret = max(ret, solve(state | (1 << v), sheep, wolf + 1, info, E))
    return ret

def solution(info, edges):
    E = [[] for _ in range(len(info))]
    for p, c in edges:
        E[p].append(c)

    return solve(1, 1, 0, info, E)