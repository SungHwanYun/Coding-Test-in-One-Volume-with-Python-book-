import sys
input=sys.stdin.readline
sys.setrecursionlimit(100000)

def solution(n):
    return solve(n)

def solve(n):
    if n == 1:
        return 1
    return n + solve(n - 1)

n = int(input())
print(solution(n))