def solution(numbers):
    answer = []
    for n in numbers:
        b = translate_binary(n)
        answer.append(solve(b, 0, len(b) - 1))
    return answer

def translate_binary(n):
    answer = []
    while n > 0:
        x = n % 2
        n = n // 2
        answer.append(x)

    y = 1
    while y <= len(answer):
        y = y * 2
    while len(answer) + 1 < y:
        answer.append(0)
    answer.reverse()
    return answer

def solve(b, st, en):
    if st == en:
        return 1

    r = (en + st) // 2

    if b[r] == 0:
        for i in range(st, en + 1):
            if b[i] == 1:
                return 0
        return 1
    else:
        return solve(b, st, r - 1) & solve(b, r + 1, en)