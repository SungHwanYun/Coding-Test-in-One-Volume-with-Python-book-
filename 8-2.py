import sys
input=sys.stdin.readline

def solution(n, A, B):
    d = {}
    for a in A.split():
        d[a] = 0

    for b in B:
        for c in b.split():
            d[c] += 1

    answer = list(d.items())
    answer.sort(key = lambda x: (-x[1], x[0]))
    return answer

n = int(input())
A = input().strip()
B = list(input().strip() for _ in range(n))
C = solution(n, A, B)
for name, value in C:
    print(name, value)