import sys
input = sys.stdin.readline
sys.setrecursionlimit(10 ** 6)

def solution(n, k, A, edges):
    E = list([] for _ in range(n))
    for p, c in edges:
        E[p].append(c)

    return solve(0, k, A, E)

def solve(u, k, A, E):
    ret = A[u]
    if k == 0:
        return ret
    for v in E[u]:
        ret += solve(v, k - 1, A, E)

    return ret

n, k = map(int, input().split())
edges = list(list(map(int, input().split())) for _ in range(n - 1))
A = list(map(int, input().split()))
print(solution(n, k, A, edges))