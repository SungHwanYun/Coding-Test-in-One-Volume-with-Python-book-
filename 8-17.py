import sys
input=sys.stdin.readline
import itertools
def solution(A):
    src = [0, 1, 2, 3, 4, 5]
    answer = int(1e8)
    for P in itertools.permutations(src):
        D = list([0] * 2 for _ in range(6))
        D[0][0] = D[0][1] = A[P[0]*2][P[0]*2+1]
        for i in range(1, 6):
            D[i][0] = min(D[i - 1][0] + A[P[i-1]*2][P[i]*2 + 1] + \
                            A[P[i]*2][P[i]*2 + 1], \
                            D[i - 1][1] + A[P[i-1]*2+1][P[i]*2 + 1] + \
                            A[P[i]*2][P[i]*2 + 1])

            D[i][1] = min(D[i - 1][0] + A[P[i-1]*2][P[i]*2] + \
                            A[P[i]*2][P[i]*2 + 1], \
                          D[i - 1][1] + A[P[i-1]*2+1][P[i]*2] + \
                            A[P[i]*2][P[i]*2 + 1])
        answer = min(answer, min(D[5][0], D[5][1]))
    return answer

A = list(list(map(int, input().split())) for _ in range(12))
print(solution(A))