import sys
input=sys.stdin.readline

def solution(n, A, k):
    answer = 0
    for a in A:
        if a == k:
            answer += 1
    return answer

n, k = map(int, input().split())
A = list(map(int, input().split()))
print(solution(n, A, k))