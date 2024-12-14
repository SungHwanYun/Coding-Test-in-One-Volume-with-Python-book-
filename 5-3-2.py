import sys
input=sys.stdin.readline

def solution(n, k):
    a = 0
    while n > 0:
        d = n % k
        n = n // k
        a += d
        
    b = ''
    while a > 0:
        d = a % k
        a = a // k
        b += str(d)
    ret = b[::-1]
    
    return int(ret)

n, k = map(int, input().split())
print(solution(n, k))