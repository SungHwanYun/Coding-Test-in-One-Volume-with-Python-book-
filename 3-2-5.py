import sys
input=sys.stdin.readline
from itertools import permutations

def solution(A):
    A=''.join(sorted(A))
    PA = []
    for p in permutations(A):
        PA.append(''.join(p))
    return PA

A = input().strip()
PA = solution(A)
for a in PA:
    print(a)