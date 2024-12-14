import sys
input=sys.stdin.readline

from itertools import combinations

def solution(A, k):
    B = list(combinations(A, k))
    
    C = list(''.join(b) for b in B)
    C.sort()
    return C
  
A = input().strip()
k = int(input().strip())
C = solution(A, k)
for c in C:
    print(c)