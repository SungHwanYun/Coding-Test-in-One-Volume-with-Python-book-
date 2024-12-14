import sys
input=sys.stdin.readline

def solution(A, B):
    D = {}
    for phone in A:
        for i in range(len(phone) - 1):
            x = phone[:i + 1]
            if x in D:
                D[x] += 1
            else:
                D[x] = 1

    if B in D:
        return D[B]
    else:
        return 0

A = list(input().split())
B = input().strip()
print(solution(A, B))