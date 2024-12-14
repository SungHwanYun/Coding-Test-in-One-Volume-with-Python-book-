import sys
input=sys.stdin.readline
sys.setrecursionlimit(10**6)

def solution(n, A, edges):
    E = list([] for _ in range(n))
    for p, c in edges:
        E[p].append(c)

    return min(solve(0, 0, A, E), solve(0, 1, A, E))

def solve(u, color, A, E):
    ret = A[u][color]
    for v in E[u]:
        ret += solve(v, 1 - color, A, E)

    return ret

n = int(input())
edges = list(list(map(int, input().split())) for _ in range(n - 1))
A = list(list(map(int, input().split())) for _ in range(n))
print(solution(n, A, edges))