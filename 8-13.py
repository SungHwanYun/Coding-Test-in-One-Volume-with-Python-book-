import sys
input = sys.stdin.readline
sys.setrecursionlimit(10 ** 5)

def solution(n, A):
    T = [0] * 24 * 60 * 60
    R = [0] * 24 * 60 * 60
    answer = 0
    for a in A:
        if a[0] == '1':
            add_query(T, translate_time(a[1]), translate_time(a[2]))
        else:
            for t in range(1, 24 * 60 * 60):
                T[t] += T[t - 1]

            R[0] = T[0]
            for t in range(1, 24 * 60 * 60):
                R[t] += R[t - 1] + T[t]

            answer = get_max_range(R, translate_time(a[1]))

    return answer

def translate_time(t):
    return int(t[:2]) * 3600 + int(t[3:5]) * 60 + int(t[6:])

def add_query(T, i, j):
    T[i] += 1
    T[j] -= 1

def get_max_range(R, range_len):
    ret = 0
    for j in range(range_len - 1, 24 * 60 * 60):
        i = j - range_len + 1
        a = R[j]
        if i != 0:
            a -= R[i - 1]
        ret = max(ret, a)
    return ret

n = int(input().strip())
A = list(list(input().split()) for _ in range(n))
print(solution(n, A))