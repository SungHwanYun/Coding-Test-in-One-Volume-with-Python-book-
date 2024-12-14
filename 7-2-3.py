import sys
input = sys.stdin.readline
sys.setrecursionlimit(10 ** 6)

def solution(n, A, edges):
    E = list([] for _ in range(n))
    for p, c in edges:
        E[p].append(c)

    return solve(0, A, E)

def solve(u, A, E):
    ret = A[u]
    for v in E[u]:
        ret2 = solve(v, A, E)
        if ret2 > 0:
            ret += ret2

    return ret

n = int(input())
edges = list(list(map(int, input().split())) for _ in range(n - 1))
A = list(map(int, input().split()))
print(solution(n, A, edges))