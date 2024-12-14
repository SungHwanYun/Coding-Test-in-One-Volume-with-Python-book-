import sys
input=sys.stdin.readline

def solution(n, m, A, B):
    answer = []
    for qry in B:
        cnt = 0
        for student in A:
            if is_ok(qry, student):
                cnt += 1
        answer.append(cnt)

    return answer

def is_ok(qry, student):
    for i in range(3):
        if qry[i] != '-'  and qry[i] != student[i]:
            return False
    return True

n, m = map(int, input().split())
A = list(list(input().split()) for _ in range(n))
B = list(list(input().split()) for _ in range(m))
C = solution(n, m, A, B)
for c in C:
    print(c)