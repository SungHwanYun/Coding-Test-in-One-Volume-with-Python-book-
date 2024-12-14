import sys
input=sys.stdin.readline

def solution(n):
    answer = 0
    for A in range(10**(n - 1), 10**n):
        if is_ok(A):
            answer += 1
    return answer

def is_ok(A):
    p = A % 10
    A //= 10
    if p == 0: 
        return False
    while A > 0:
        c = A % 10
        A //= 10
        
        if c == 0 or abs(p - c) > 2:
            return False
        
        p = c
        
    return True
    
n = int(input().strip())
print(solution(n))