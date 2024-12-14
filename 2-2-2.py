import sys
input=sys.stdin.readline

def solution(A):
    B = ''
    for a in A:
        if a.islower():
            B += a

    return B

A = input().strip()
print(solution(A))