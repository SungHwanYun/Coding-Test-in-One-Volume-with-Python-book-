import sys
input = sys.stdin.readline
sys.setrecursionlimit(10 ** 5)
def solution(n, A):
    T = [0] * 24 * 60 * 60
    R = [0] * 24 * 60 * 60
    answer = []
    flag = False
    for a in A:
        if a[0] == '1':
            add_query(T, translate_time(a[1]), translate_time(a[2]))
        else:
            if flag == False:
                for t in range(1, 24 * 60 * 60):
                    T[t] += T[t - 1]

                flag = True
                R[0] = T[0]
                for t in range(1, 24 * 60 * 60):
                    R[t] += R[t - 1] + T[t]

            ret = get_sum(R, translate_time(a[1]), translate_time(a[2]))
            answer.append(ret)

    return answer

def translate_time(t):
    return int(t[:2]) * 3600 + int(t[3:5]) * 60 + int(t[6:])

def add_query(T, i, j):
    T[i] += 1
    T[j] -= 1

def get_sum(R, i, j):
    ret = R[j - 1]
    if i != 0:
        ret -= R[i - 1]
    return ret

n = int(input().strip())
A = list(list(input().split()) for _ in range(n))
B = solution(n, A)
for b in B:
    print(b)