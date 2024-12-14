import sys
input=sys.stdin.readline

def solution(A):
    B = A[1::2]
    return B

A = input().strip()
print(solution(A))