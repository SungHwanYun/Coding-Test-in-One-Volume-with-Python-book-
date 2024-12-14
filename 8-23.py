import sys
input=sys.stdin.readline
sys.setrecursionlimit(10**5)

def solution(n, k, A, edges):
    E = list([] for _ in range(n))
    for p, c in edges:
        E[p].append(c)

    visited = [0] * (1<<n)
    return solve(1<<0, k - int(A[0] == 1), int(A[0] == 2), A, E, visited)

def solve(state, k, pear, A, E, visited):
    if visited[state] == 1:
        return 0
    visited[state]=1

    ret = pear

    for u in range(len(A)):
        if (state & (1 << u)) == 0:
            continue

        for v in E[u]:
            if state & (1 << v):
               continue

            if A[v] == 1 and k == 0:
                continue

            ret = max(ret, solve(state | (1 << v), k - int(A[v] == 1), \
                         pear + int(A[v] == 2), A, E, visited))

    return ret

n, k = map(int, input().split())
edges = list(list(map(int, input().split())) for _ in range(n - 1))
A = list(map(int, input().split()))
print(solution(n, k, A, edges))