import sys
input=sys.stdin.readline

def solution(n, k, A):
    B = [0] * (n+1)
    for a in A:
        B[a] = 1

    D = [0] * (n + 2)
    for i in range(n, 0, -1):
        for j in range(i, i + k, 1):
            if j > n: break

            if B[j] == 1:
                continue

            if D[j + 1] == 0:
                D[i] = 1
                break

    return D[1]

n, k = map(int, input().split())
A = list(map(int, input().split()))
print(solution(n, k, A))