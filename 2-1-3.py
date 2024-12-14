import sys
input=sys.stdin.readline

def solution(A, B):
    a, b = 0, 0
    for x, y in zip(A, B):
        if x > y:
            a += 1
        elif x < y:
            b += 1
    return int(a > b)

A = map(int, input().split())
B = map(int, input().split())
print(solution(A, B))