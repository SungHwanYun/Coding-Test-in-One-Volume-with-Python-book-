import sys
input=sys.stdin.readline

# n, A: 원소의 개수가 n인 정수형 배열 A
# i, j, k: i부터 j까지의 배열 A의 원소에 k를 곱하는 연산 수행
# 연산을 수행한 후 배열 A의 원소의 합을 반환한다.
def solution(n, A, i, j, k):
    # i부터 j까지의 배열 A의 원소에 k를 곱하는 연산을 수행한다.
    for idx in range(i, j + 1):
        A[idx] = A[idx] * k
        
    return sum(A)

# 입력을 받는다
n = map(int, input().split())
A = list(map(int, input().split()))
i, j, k = map(int, input().split())
print(solution(n, A, i, j, k))