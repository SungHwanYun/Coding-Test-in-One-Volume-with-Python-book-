import sys
input=sys.stdin.readline

def solution(n):
    if n <= 2: return 1
    d = [0] * (n + 1)
    d[1] = d[2] = 1
    for i in range(3, n + 1):
        d[i] = (d[i - 1] + d[i - 2]) % 987654321
    return d[n]

n = int(input())
print(solution(n))