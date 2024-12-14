A = [0] * 360004
B = [0] * 360004
P = T = None

def convert_time(str):
    h = int(str[0:2]); m = int(str[3:5]); s = int(str[6:8])
    return h * 3600 + m * 60 + s

def solution(play_time, adv_time, logs):
    P = convert_time(play_time)
    T = convert_time(adv_time)

    for log in logs:
        s = convert_time(log)
        e = convert_time(log[9:])

        A[s] +=1 ; A[e] -= 1

    for i in range(1, P + 1, 1):
        A[i] += A[i - 1]

    B[0] = A[0]
    for i in range(1, P + 1, 1):
        B[i] = B[i - 1] + A[i]

    x = 0; y = B[T - 1]
    i = 1
    while i + T <= P:
        sum = B[i + T - 1] - B[i - 1]
        if sum > y:
            y = sum
            x = i
        i += 1

    h = x // 3600
    m = (x - h * 3600) // 60
    s = x % 60
    answer = '%02d:%02d:%02d' % (h, m, s)
    return answer