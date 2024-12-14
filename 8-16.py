import sys

input = sys.stdin.readline


def solution(A):
    D = list([0] * 2 for _ in range(6))
    D[0][0] = D[0][1] = A[0][1]
    for i in range(1, 6):
        D[i][0] = min(D[i - 1][0] + A[2 * i - 2][2 * i + 1] + A[2 * i][2 * i + 1], \
                      D[i - 1][1] + A[2 * i - 1][2 * i + 1] + A[2 * i][2 * i + 1])

        D[i][1] = min(D[i - 1][0] + A[2 * i - 2][2 * i] + A[2 * i][2 * i + 1], \
                      D[i - 1][1] + A[2 * i - 1][2 * i] + A[2 * i][2 * i + 1])

    return min(D[5][0], D[5][1])


A = list(list(map(int, input().split())) for _ in range(12))
print(solution(A))