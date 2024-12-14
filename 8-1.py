import sys
input=sys.stdin.readline

def solution(n, A, fees):
    d = {}
    for t, name in map(parse_log, A):
        if name in d:
            d[name] += t
        else:
            d[name] = t

    for key, value in d.items():
        d[key] = get_fee(fees, value)

    answer = list(d.items())
    answer.sort(key = lambda x : (-x[1], x[0]))
    return answer

def parse_log(s):
    t = int(s[0:2]) * 60 + int(s[3:5])
    name = s[6:]
    return [t, name]

def get_fee(fees, t):
    money = fees[1]
    if fees[0] < t:
        money += (t - fees[0] + fees[2] - 1) // fees[2] * fees[3]

    return money

n = int(input())
A = list(input().rstrip() for _ in range(n))
fees = [100, 10, 50, 3]
B = solution(n, A, fees)
for name, cost in B:
    print(name, cost)