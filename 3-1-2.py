import sys
input=sys.stdin.readline

def solution(n, m, A, B):
    answer = []
    for subject in B:
        if subject == '-':
            answer.append(n)
        else:
            cnt = 0
            for a in A:
                if a == subject:
                    cnt += 1
            answer.append(cnt)

    return answer

n, m = map(int, input().split())
A = list(input().split())
B = list(input().strip() for _ in range(m))
C = solution(n, m, A, B)
for c in C:
    print(c)