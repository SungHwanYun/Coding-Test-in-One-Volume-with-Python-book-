def solution(n, A):
    mx = 0
    D = {}
    for a in A:
        if a in D:
            D[a] += 1
        else:
            D[a] = 1
        mx = max(mx, D[a])

    answer = []
    for key, value in D.items():
        if value == mx:
            answer.append(key)
    answer.sort()
    return answer

n = int(input())
A = list(map(int, input().split()))
B = solution(n, A)
for b in B:
    print(b, end=' ')