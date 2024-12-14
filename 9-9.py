def is_prime(x):
    if x <= 1:
        return False
    for i in range(2, x + 1):
        if i * i > x:
            return True
        if x % i == 0:
            return False

def solution(n, k):
    answer = 0

    P = ""
    while n > 0:
        d = n % k
        n = n // k
        P += str(d)
    P = P[ : : -1]

    for x in P.split('0'):
        if x != '' and is_prime(int(x)):
            answer += 1

    return answer