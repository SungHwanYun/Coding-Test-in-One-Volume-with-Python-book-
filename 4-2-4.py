import sys
input=sys.stdin.readline

def solution(n):
    return solve([], n)

def solve(A, n):
    m = len(A)

    if m == n:
        return 1
    
    ret = 0
    if m == 0:
        s, e = 1, 9
    else:
        s = max(A[m - 1] - 2, 1)
        e = min(A[m - 1] + 2, 9)
        
    for a in range(s, e + 1):
        A.append(a)
        ret += solve(A, n)
        A.pop()
        
    return ret
    
n = int(input().strip())
print(solution(n))