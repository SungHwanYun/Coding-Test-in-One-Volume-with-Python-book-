def translate_days(x):
    return int(x[:4]) * 12 * 28 + int(x[5:7]) * 28 + int(x[8:])

def solution(today, terms, privacies):
    T = {}
    for t in terms:
        x, y = t.split()
        T[x] = int(y) * 28

    answer = []

    today = translate_days(today)

    for i in range(len(privacies)):
        x, y = privacies[i].split()

        a = translate_days(x) + T[y]
        if a <= today:
            answer.append(i + 1)

    return answer