import sys
input = sys.stdin.readline
def solution(n, A):
    A.sort()
    B = []
    return solve(n, A, B)
def solve(n, A, B):
    ret = [-1]

    if n == 0:
        pa, pb = P(A), P(B)
        if pa < pb:
            return B
        return [-1]

    start = 1
    if len(B) != 0: start = B[-1]
    for card in range(start, 10):
        B.append(card)
        ret2 = solve(n - 1, A, B)
        if ret2[0] != -1:
            if ret[0] == -1:
                ret = [0] * n
                ret[:] = ret2[:]
            else:
                ret_num = get_joined_num(ret)
                ret2_num = get_joined_num(ret2)

                if ret_num > ret2_num:
                    ret[:] = ret2[:]
        B.pop()

    return ret

def P(X):
    ret = 1
    for x in X:
        ret *= x
    return ret

def get_joined_num(X):
    x = ''.join(map(str, X))
    return int(x)

n = int(input())
A = list(map(int, input().split()))
B = solution(n, A)
for b in B: print(b, end=' ')