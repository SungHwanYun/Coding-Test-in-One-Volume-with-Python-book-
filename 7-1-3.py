import sys
input=sys.stdin.readline

def solution(n):
    D = list([0] * 10 for _ in range(n + 1))
    for j in range(1, 10):
        D[1][j] = 1

    for i in range(2, n + 1):
        for j in range(1, 10):
            s = max(j - 2, 1)
            e = min(j + 2, 9)

            for k in range(s, e + 1):
                D[i][j] += D[i - 1][k]
            D[i][j] %= 987654321

    answer = 0
    for j in range(1, 10):
        answer += D[n][j]
    return answer % 987654321

n = int(input())
print(solution(n))