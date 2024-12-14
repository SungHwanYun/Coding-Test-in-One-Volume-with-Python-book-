import sys
input=sys.stdin.readline
sys.setrecursionlimit(100000)

def solution(n):
    return solve(n)

def solve(n):
    if n <= 2:
        return 1
    return solve(n - 1) + solve(n - 2)

n = int(input())
print(solution(n))