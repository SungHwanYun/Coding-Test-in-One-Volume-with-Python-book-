import sys
input=sys.stdin.readline

def solution(n, k, A):
    B = [0] * (n+1)
    for a in A:
        B[a]=1

    return abs(solve(n, k, 0, B))

def solve(n, k, a, B):
    if a == n:
        return 0

    nxt = []
    for b in range(a + 1, a + k + 1):
        if b > n:
            break

        if B[b]==1:
            continue

        nxt.append(solve(n, k, b, B))

    if len(nxt) == 0:
        return 0

    nxt.sort()

    if nxt[0] > 0:
        return -(nxt[-1] + 1)

    ret = None
    for i in range(len(nxt)):
        if nxt[i] <= 0:
            ret = nxt[i]
    return -ret + 1

n, k = map(int, input().split())
A = list(map(int, input().split()))
print(solution(n, k, A))