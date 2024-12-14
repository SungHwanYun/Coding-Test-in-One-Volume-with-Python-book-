import sys
input=sys.stdin.readline

def solution(A, k):
    while len(A) < k:
        A += A[-1]
    return A

def solution2(A, k):
    A = A + A[-1]*(k - len(A))
    return A

A = input().strip()
k = int(input())
print(solution(A, k))