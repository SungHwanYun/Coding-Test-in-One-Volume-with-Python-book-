import sys
input=sys.stdin.readline

def solution(n, A, m, Q):
    for q in Q:
        if q[0] == 1:
            do_add_query(A, q[1], q[2], q[3], q[4], q[5])
        else:
            print(get_sum(A, q[1], q[2], q[3], q[4]))

def do_add_query(A, i1, j1, i2, j2, k):
    for i in range(i1, i2 + 1, 1):
        for j in range(j1, j2 + 1, 1):
            A[i][j] += k

def get_sum(A, i1, j1, i2, j2):
    ret = 0
    for i in range(i1, i2 + 1, 1):
        for j in range(j1, j2 + 1, 1):
            ret += A[i][j]
    return ret
    
n, m = map(int, input().split())
A = list(list(map(int, input().split())) for _ in range(n))
Q = list(list(map(int, input().split())) for _ in range(m))
solution(n, A, m, Q)