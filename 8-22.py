import sys

input = sys.stdin.readline
sys.setrecursionlimit(10 ** 5)


def solution(n, k, A, edges):
    E = list([] for _ in range(n))
    for p, c in edges:
        E[p].append(c)

    visited = [0] * (1 << n)
    return solve(1 << 0, k - 1, int(A[0] == 1), int(A[0] == 2), A, E, visited)


def solve(state, k, apple, pear, A, E, visited):
    if visited[state] == 1:
        return [0, 0]
    visited[state] = 1

    ret = [apple, pear]

    if k == 0:
        return ret

    for u in range(len(A)):
        if (state & (1 << u)) == 0:
            continue

        for v in E[u]:
            if state & (1 << v):
                continue

            ret2 = solve(state | (1 << v), k - 1, \
                         apple + int(A[v] == 1), pear + int(A[v] == 2), A, E, visited)

            if ret2[0] * ret2[1] > ret[0] * ret[1] or \
                    (ret2[0] * ret2[1] == ret[0] * ret[1] and ret2[0] > ret[0]) or \
                    (ret2[0] * ret2[1] == ret[0] * ret[1] and ret2[0] == ret[0] and ret2[1] > ret[1]):
                ret[0], ret[1] = ret2[0], ret2[1]

    return ret


n, k = map(int, input().split())
edges = list(list(map(int, input().split())) for _ in range(n - 1))
A = list(map(int, input().split()))
ret = solution(n, k, A, edges)
print(ret[0], ret[1])