import sys
input=sys.stdin.readline

def solution(n, k):
    a = ''
    while n > 0:
        d = n % k
        n = n // k
        a += str(d)
    a = a[::-1]

    c = 0
    for b in a.split('0'):
        if b != '':
            c += int(b)

    ret = ''
    while c > 0:
        d = c % k
        c = c // k
        ret += str(d)
    ret = ret[::-1]

    return int(ret)

n, k = map(int, input().split())
print(solution(n, k))