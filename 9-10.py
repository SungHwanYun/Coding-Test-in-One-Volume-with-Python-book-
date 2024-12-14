def parse_log(s):
    t = int(s[0:2]) * 60 + int(s[3:5])

    c = int(s[6:10])

    if s[11:13] == 'IN':
        return [t, c, 0]
    else:
        return [t, c, 1]

def get_fee(fees, t):
    money = fees[1]

    if fees[0] < t:
        money += (t - fees[0] + fees[2] - 1) // fees[2] * fees[3]

    return money

def solution(fees, records):
    answer = []
    parking_time = [0] * 10000
    in_time = [-1] * 10000

    for t, c, d in map(parse_log, records):
        if d == 0:
            in_time[c] = t
        else:  
            parking_time[c] += t - in_time[c]

            in_time[c] = -1

    for i in range(10000):
        if in_time[i] != -1: 
            parking_time[i] += 23 * 60 + 59 - in_time[i]

    for i in range(10000):
        if parking_time[i] != 0:
            answer.append(get_fee(fees, parking_time[i]))

    return answer