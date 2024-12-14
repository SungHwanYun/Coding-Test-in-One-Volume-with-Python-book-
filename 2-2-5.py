import sys
input=sys.stdin.readline

def parse_log(s):
    return int(s[0:2]) * 60 + int(s[3:5])

def solution(records):
    total_cost = 0
    
    for t in map(parse_log, records):
        total_cost += t

    hour = total_cost // 60
    minute = total_cost % 60
    
    ret = ''
    if hour < 100:
        ret = '%02d:%02d'%(hour, minute)
    else:
        ret = '%d:%02d'%(hour, minute)
    return ret

records = list(input().split())
print(solution(records))