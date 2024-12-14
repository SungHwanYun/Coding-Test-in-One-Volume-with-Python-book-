import sys
input=sys.stdin.readline

from bisect import bisect_left, bisect_right
def solution2(n, m, A, B):
    A.sort()
    
    answer = []
    for k1, k2 in B:
        i, j = bisect_left(A, k1), bisect_right(A, k2)
        answer.append(j - i)
        
    return answer
        
n, m = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()) for _ in range(m))
C = solution2(n, m, A, B)
for c in C:
    print(c)