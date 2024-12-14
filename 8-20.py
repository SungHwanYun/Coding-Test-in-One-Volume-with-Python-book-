import sys
sys.setrecursionlimit(10 ** 6)

def solution(n, A, edges):
    E = list([] for _ in range(n))
    for p, c in edges:
        E[p].append(c)

    D = [[-1] * 2 for _ in range(n)]

    return min(solve(0, 0, A, E, D), solve(0, 1, A, E, D))

def solve(u, color, A, E, D):
    if D[u][color] != -1: return D[u][color]

    D[u][color] = A[u][color]
    for v in E[u]:
        if color == 0:
            D[u][color] += min(solve(v, 0, A, E, D), solve(v, 1, A, E, D))
        else:
            D[u][color] += solve(v, 0, A, E, D)

    return D[u][color]

n = int(input())
edges = list(list(map(int, input().split())) for _ in range(n - 1))
A = list(list(map(int, input().split())) for _ in range(n))
print(solution(n, A, edges))