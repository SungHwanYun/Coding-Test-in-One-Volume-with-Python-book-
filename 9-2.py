def solution(cap, n, deliveries, pickups):
    i = n - 1
    j = n - 1
    answer = 0
    while i >= 0 or j >=0:
        x = cap
        ii , jj = -1, -1
        while i >= 0 and x > 0:
            if ii == -1 and deliveries[i] > 0:
                ii = i
            y = min(deliveries[i], x)
            x -= y
            deliveries[i] -= y
            if deliveries[i] == 0:
                i -= 1

        x = cap
        while j >= 0 and x > 0:
            if jj == -1 and pickups[j] > 0:
                jj = j
            y = min(pickups[j], x)
            x -= y
            pickups[j] -= y
            if pickups[j] == 0:
                j -= 1

        if ii == -1 and jj == -1:
            break

        answer += max(ii + 1, jj + 1) * 2

    return answer