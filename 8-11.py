import sys
input = sys.stdin.readline
sys.setrecursionlimit(10 ** 5)

def solution(n, A, m, Q):
    psum = list([0] * n for _ in range(n))
    psum_flag = False

    for q in Q:
        if q[0] == 1:
            do_add_query(psum, q[1], q[2], q[3], q[4], q[5])
        else:
            if psum_flag == False:
                psum_flag = True
                for r in range(n):
                    for c in range(1, n, 1):
                        psum[r][c] += psum[r][c - 1]

                for c in range(n):
                    for r in range(1, n, 1):
                        psum[r][c] += psum[r - 1][c]

                for r in range(n):
                    for c in range(n):
                        A[r][c] += psum[r][c]

                psum[0][0] = A[0][0]
                for c in range(1, n, 1):
                    psum[0][c] = psum[0][c - 1] + A[0][c]
                for r in range(1, n, 1):
                    psum[r][0] = psum[r - 1][0] + A[r][0]
                for r in range(1, n, 1):
                    for c in range(1, n, 1):
                        psum[r][c] = psum[r - 1][c] + psum[r][c - 1] - psum[r - 1][c - 1] + A[r][c]

            print(get_sum(psum, q[1], q[2], q[3], q[4]))

def do_add_query(A, i1, j1, i2, j2, k):
    A[i1][j1] += k
    if j2 + 1 < n:
        A[i1][j2 + 1] -= k
    if i2 + 1 < n:
        A[i2 + 1][j1] -= k
    if i2 + 1 < n and j2 + 1 < n:
        A[i2 + 1][j2 + 1] += k

def get_sum(psum, i1, j1, i2, j2):
    ret = psum[i2][j2]
    if i1 > 0:
        ret -= psum[i1 - 1][j2]
    if j1 > 0:
        ret -= psum[i2][j1 - 1]
    if i1 > 0 and j1 > 0:
        ret += psum[i1 - 1][j1 - 1]
    return ret

n, m = map(int, input().split())
A = list(list(map(int, input().split())) for _ in range(n))
Q = list(list(map(int, input().split())) for _ in range(m))
solution(n, A, m, Q)