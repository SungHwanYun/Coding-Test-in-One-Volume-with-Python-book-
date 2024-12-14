import sys
input=sys.stdin.readline
sys.setrecursionlimit(100000)

def solution(A):
    solve(A, 0)

def solve(A, B):
    if A != 0:
        if A%10 != 0 or B == 1:
            print(A%10, end='')
            B = 1
        solve(A // 10, B)

A = int(input().strip())
solution(A)