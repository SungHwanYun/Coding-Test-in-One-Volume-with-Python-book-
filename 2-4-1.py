from collections import deque
def solution(n, A):
    answer = [0, 0]
    q = deque()
    for info in A:
        if info[0] == 1:
            q.append(info[1])
            if answer[0] < len(q) or \
                    (answer[0] == len(q) and answer[1] > info[1]):
                answer = [len(q), info[1]]
        else:
            q.popleft()
    return answer

n = int(input())
A = list(list(map(int, input().split())) for _ in range(n))
B = solution(n, A)
print(B[0], B[1])