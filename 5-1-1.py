import sys
input=sys.stdin.readline

def solution(n, A):
    T = [0] * 24*60*60
    answer = 0
    for a in A:
        if a[0] == '1':
            add_query(T, translate_time(a[1]), translate_time(a[2]))
        else:
            answer = get_sum(T, translate_time(a[1]), translate_time(a[2]))

    return answer

def translate_time(t):
    return int(t[:2])*3600 + int(t[3:5])*60 + int(t[6:])

def add_query(T, i, j):
    T[i] += 1
    T[j] -= 1

def get_sum(T, i, j):
    for t in range(1, 24*60*60):
        T[t] += T[t - 1]
        
    ret = 0
    for t in range(i, j):
        ret += T[t]
    return ret
    
n = int(input().strip())
A = list(list(input().split()) for _ in range(n))
print(solution(n, A))