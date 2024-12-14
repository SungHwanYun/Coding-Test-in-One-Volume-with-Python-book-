import sys

input = sys.stdin.readline
from itertools import combinations


def solution(X, Y, Z, k):
    CX = list(combinations(X, k))
    CX = list(''.join(x) for x in CX)
    CY = list(combinations(Y, k))
    CY = list(''.join(y) for y in CY)
    CZ = list(combinations(Z, k))
    CZ = list(''.join(z) for z in CZ)

    d = {}
    solve(CX, d)
    solve(CY, d)
    solve(CZ, d)

    answer = []
    for key, value in d.items():
        if value >= 2:
            answer.append(key)
    answer.sort()

    if len(answer) == 0:
        answer = ['-1']

    return answer


def solve(C, d):
    for c in C:
        if c in d:
            d[c] += 1
        else:
            d[c] = 1


X = input().strip()
Y = input().strip()
Z = input().strip()
k = int(input())
C = solution(X, Y, Z, k)
for c in C:
    print(c)