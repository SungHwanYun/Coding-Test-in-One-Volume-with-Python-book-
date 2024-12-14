import sys
input=sys.stdin.readline

def solution(A, B):
    for b in B:
        A = A.replace(b, b.lower())

    return A

A = input().strip()
B = list(map(str, input().split()))
print(solution(A, B))