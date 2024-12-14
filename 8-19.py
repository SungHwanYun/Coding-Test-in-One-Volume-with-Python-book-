import sys

input = sys.stdin.readline


def solution(n, k, A):
    B = [0] * (n + 1)
    for a in A:
        B[a] = 1

    D = [0] * (n + 2)
    for i in range(n, 0, -1):
        nxt = []
        for j in range(i, i + k, 1):
            if j > n: break

            if B[j] == 1:
                continue

            nxt.append(D[j + 1])

        if len(nxt) == 0:
            D[i] = 0
            continue

        nxt.sort()

        if nxt[0] > 0:
            D[i] = -(nxt[-1] + 1)
            continue

        ret = None
        for p in range(len(nxt)):
            if nxt[p] <= 0:
                ret = nxt[p]
        D[i] = -ret + 1

    return abs(D[1])


n, k = map(int, input().split())
A = list(map(int, input().split()))
print(solution(n, k, A))