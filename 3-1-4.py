import sys
input=sys.stdin.readline

def solution(n, m, A, B):
    for b in B:
        if b[0] == 1:
            do_add_query(A, b[1], b[2], b[3])
        else:
            print(sum(A[b[1]:b[2] + 1]))

def do_add_query(A, i, j, k):
    for p in range(i, j + 1, 1):
        A[p] += k

n, m = map(int, input().split())
A = list(map(int, input().split()))
B = list(list(map(int, input().split())) for _ in range(m))
solution(n, m, A, B)