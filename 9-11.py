def get_point(rian, info):
    rian_point = apeach_point = 0

    for i in range(1, 11, 1):
        if rian[i] == 0 and info[i] == 0:
            continue

        if rian[i] > info[i]:
            rian_point += i
        else:
            apeach_point += i

    if rian_point > apeach_point:
        return rian_point - apeach_point
    else:
        return -1

def solve(rian, n, info):
    arrow = [-1]
    point = 0
    
    if len(rian) == 10:
        rian.append(n - sum(rian))
        x = get_point(rian, info)

        if x > point:
            point = x
            arrow = rian[:]
        rian.pop()
        return point, arrow

    for i in range(n - sum(rian), -1, -1):
        rian.append(i)
        p, a = solve(rian, n, info)
        if (point < p):
            point, arrow = p, a
        rian.pop()
        
    return point, arrow

def solution(n, info):
    info.reverse()

    rian = []

    p, a = solve(rian, n, info)
    return list(reversed(a))