import sys
input=sys.stdin.readline

def solution(n, A, m, Q):
    psum = [0] * n
    
    for q in Q:
        if q[0] == 1:
            do_add_query(psum, q[1], q[2], q[3])
        else:
            for i in range(1, n, 1):
                psum[i] += psum[i - 1]
            print(sum(psum[q[1]:q[2] + 1]) + sum(A[q[1]:q[2] + 1]))

def do_add_query(psum, i, j, k):
    psum[i] += k
    if j + 1 < n:
        psum[j + 1] -= k
    
n, m = map(int, input().split())
A = list(map(int, input().split()))
Q = list(list(map(int, input().split())) for _ in range(m))
solution(n, A, m, Q)