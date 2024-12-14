import sys

input = sys.stdin.readline


def solution(n, k, A):
    B = [0] * (n + 1)
    for a in A:
        B[a] = 1

    return solve(n, k, 0, B)


def solve(n, k, a, B):
    if a == n:
        return 0

    for b in range(a + 1, a + k + 1):
        if b > n:
            break

        if B[b] == 1:
            continue

        if solve(n, k, b, B) == 0:
            return 1

    return 0


n, k = map(int, input().split())
A = list(map(int, input().split()))
print(solution(n, k, A))