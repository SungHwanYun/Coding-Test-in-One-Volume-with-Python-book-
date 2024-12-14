import sys
input = sys.stdin.readline
sys.setrecursionlimit(10 ** 6)

def solution(n, k, A, edges):
    E = list([] for _ in range(n))
    for p, c in edges:
        E[p].append(c)
    return solve(0, 0, k, A, E)

def solve(u, depth, k, A, E):
    if A[u] == k:
        return depth
    for v in E[u]:
        ret = solve(v, depth + 1, k, A, E)
        if ret != -1:
            return ret
    return -1

n, k = map(int, input().split())
edges = list(list(map(int, input().split())) for _ in range(n - 1))
A = list(map(int, input().rstrip().split()))
print(solution(n, k, A, edges))