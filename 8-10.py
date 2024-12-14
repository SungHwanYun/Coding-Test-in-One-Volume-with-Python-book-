import sys

input = sys.stdin.readline
sys.setrecursionlimit(10 ** 5)


def solution(n, A, m, Q):
    psum = list([0] * n for _ in range(n))

    for q in Q:
        if q[0] == 1:
            do_add_query(psum, q[1], q[2], q[3], q[4], q[5])
        else:
            for r in range(n):
                for c in range(1, n):
                    psum[r][c] += psum[r][c - 1]

            for c in range(n):
                for r in range(1, n, ):
                    psum[r][c] += psum[r - 1][c]

            print(get_sum(A, q[1], q[2], q[3], q[4]) + \
                  get_sum(psum, q[1], q[2], q[3], q[4]))


def do_add_query(A, i1, j1, i2, j2, k):
    A[i1][j1] += k
    if j2 + 1 < n:
        A[i1][j2 + 1] -= k
    if i2 + 1 < n:
        A[i2 + 1][j1] -= k
    if i2 + 1 < n and j2 + 1 < n:
        A[i2 + 1][j2 + 1] += k


def get_sum(A, i1, j1, i2, j2):
    ret = 0
    for i in range(i1, i2 + 1):
        for j in range(j1, j2 + 1):
            ret += A[i][j]
    return ret


n, m = map(int, input().split())
A = list(list(map(int, input().split())) for _ in range(n))
Q = list(list(map(int, input().split())) for _ in range(m))
solution(n, A, m, Q)