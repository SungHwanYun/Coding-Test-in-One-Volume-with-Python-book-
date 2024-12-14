E = [[] for _ in range(300004)]

D = [[0]*2 for _ in range(300004)]

def solve(r, sales):
    child_sum = 0
    diff_mn = int(2e9)
    is_zero_larger = 0

    for c in E[r]:
        solve(c, sales)

        child_sum += min(D[c][0], D[c][1])

        if D[c][0] >= D[c][1]:
            is_zero_larger = 1

        if D[c][0] <= D[c][1]:
            diff_mn = min(diff_mn, D[c][1] - D[c][0])

    D[r][1] = child_sum + sales[r - 1]

    if len(E[r]) == 0:
        D[r][0] = 0
    elif is_zero_larger == 1:
        D[r][0] = child_sum
    else:
        D[r][0] = child_sum + diff_mn

def solution(sales, links):
    for p, c in links:
        E[p].append(c)

    solve(1, sales)

    return min(D[1][0], D[1][1])