import sys
input=sys.stdin.readline

def solution(n, m, A, B):
    answer = []
    for k in B:
        cnt = 0
        for a in A:
            if a >= k:
                cnt += 1
        answer.append(cnt)
        
    return answer

n, m = map(int, input().split())
A = list(map(int, input().split()))
B = list(int(input().strip()) for _ in range(m))
C = solution(n, m, A, B)
for c in C:
    print(c)