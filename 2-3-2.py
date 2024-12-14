import sys
input=sys.stdin.readline

def solution(S):    
    d = {}
    for s in S.split():
        if s in d:
            d[s] += 1
        else:
            d[s] = 1

    ret = list(d.items())
    ret.sort(key = lambda x: x[0])
    return ret

S = input().strip()
A = solution(S)
for name, value in A:
    print(name, value)