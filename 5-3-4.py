import sys
input=sys.stdin.readline

def parse_log(s):
    return int(s[0:2]) * 60 + int(s[3:5])

def get_fee(fees, t):
    money = fees[1]
    if fees[0] < t:
        money += (t - fees[0] + fees[2] - 1) // fees[2] * fees[3]

    return money

def solution(fees, records):
    total_cost = 0
    
    for t in map(parse_log, records):
        total_cost += get_fee(fees, t)

    return total_cost

records = list(input().split())
fees = [100, 10, 50, 3]
print(solution(fees, records))