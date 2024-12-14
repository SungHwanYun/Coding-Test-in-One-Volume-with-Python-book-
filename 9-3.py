def get_info(user, emoticons, x):
    m = len(emoticons)
    money = 0
    for i in range(m):
        if x[i] >= user[0]:
            money += emoticons[i] * (100 - x[i]) // 100

    if money >= user[1]:
        return [1, 0]
    else:
        return [0, money]

def solution(users, emoticons):
    n = len(users)
    m = len(emoticons)

    answer = [0, 0]

    for k in range(2**(m*2)):
        x = []
        for i in range(m):
            a = (k >> (i*2)) & 0x3
            x.append((a+1)*10)

        ans = [0, 0]
        for u in users:
            ret = get_info(u, emoticons, x)
            ans[0] += ret[0]
            ans[1] += ret[1]

        if answer[0] < ans[0] or (answer[0] == ans[0] and answer[1] < ans[1]):
            answer[0] = ans[0]
            answer[1] = ans[1]

    return answer