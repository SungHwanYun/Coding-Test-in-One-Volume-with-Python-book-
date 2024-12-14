import sys
input=sys.stdin.readline

def solution(A):
    prime_sum = 0
    for a in A:
        if is_prime(a):
            prime_sum += a
            
    return prime_sum

def is_prime(a):
    if a < 2:
        return False
    
    i = 2
    while i * i <= a:
        if a % i == 0:
            return False
        i += 1

    return True

A = list(map(int, input().split()))
print(solution(A))