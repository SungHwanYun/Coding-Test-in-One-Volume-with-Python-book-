import sys
input=sys.stdin.readline

def solution(n, m, A, B):
    A.sort()
    
    answer = []
    for k in B:
        if k <= A[0]:
            answer.append(n)
            continue
        elif A[n-1] < k:
            answer.append(0)
            continue
        
        lo = 0
        hi = n - 1
        while lo < hi:
            mid = (lo + hi) // 2
            
            if k <= A[mid]:
                hi = mid
            else:
                lo = mid + 1
        
        answer.append(n - hi)
        
    return answer

from bisect import bisect_left
def solution2(n, m, A, B):
    A.sort()
    
    answer = []
    for k in B:
        i = bisect_left(A, k)
        answer.append(n - i)
        
    return answer
        
n, m = map(int, input().split())
A = list(map(int, input().split()))
B = list(int(input().strip()) for _ in range(m))
C = solution(n, m, A, B)
for c in C:
    print(c)