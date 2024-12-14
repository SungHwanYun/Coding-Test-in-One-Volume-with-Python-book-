import sys
input=sys.stdin.readline

def solution(n, k):
    b = 0
    while n > 0:
        d = n % k
        n = n // k
        
        b = b * k + d
        
    return b

n, k = map(int, input().split())
print(solution(n, k))