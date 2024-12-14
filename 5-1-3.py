import sys
input=sys.stdin.readline

def solution(n, A, m, B):
    psum = [0] * n
    psum2 = [0] * n
    psum_flag = False
    
    for b in B:
        if b[0] == 1:
            do_add_query(psum, b[1], b[2], b[3])
        else:
            if psum_flag == False:
                psum_flag = True
                for i in range(1, n):
                    psum[i] += psum[i - 1]
                for i in range(n):
                    A[i] = A[i] + psum[i]
                psum2[0] = A[0]
                for i in range(1, n):
                    psum2[i] = psum2[i - 1] + A[i]
            if b[1] == 0:
                print(psum2[b[2]])
            else:
                print(psum2[b[2]] - psum2[b[1] - 1])

def do_add_query(psum, i, j, k):
    psum[i] += k
    if j + 1 < n:
        psum[j + 1] -= k
    
n, m = map(int, input().split())
A = list(map(int, input().split()))
B = list(list(map(int, input().split())) for _ in range(m))
solution(n, A, m, B)