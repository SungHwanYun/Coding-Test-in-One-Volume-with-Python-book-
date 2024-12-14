import sys
input=sys.stdin.readline

def solution(A, B):
    D = {}
    for b in B:
        if b in D:
            D[b] += 1
        else:
            D[b] = 1

    answer = []
    for a in A:
        if a not in D:
            answer.append(a)
    answer.sort()
    return answer

A = list(input().split())
B = list(input().split())
C = solution(A, B)
for c in C:
    print(c)