from collections import deque

def solution(n, S):
    answer = [[ ] for _ in range(3)]
    q = deque()
    for info in S:
        if info[0] == 1:
            q.append((info[1], info[2]))
        else:
            a, b = q.popleft()
            if b == info[1]:
                answer[0].append(a)
            else:
                answer[1].append(a)

    while len(q) > 0:
        a, b = q.popleft()
        answer[2].append(a)

    for i in range(3):
        answer[i].sort()
    return answer

n = int(input())
S = list(list(map(int, input().split())) for _ in range(n))
T = solution(n, S)
for t in T:
    if len(t) == 0:
        print('None')
    else:
        for x in t:
            print(x, end=' ')
        print()