import sys
input=sys.stdin.readline
def solution(A, K):
    D = [0] * (K + 1)
    for i in range(A + 1, K + 1):
        D[i] = D[i - 1] + 1

        if i % 2 == 0 and (i // 2) >= A:
            D[i] = min(D[i], D[i // 2] + 1)
    return D[K]

A, K = map(int, input().split())
print(solution(A, K))