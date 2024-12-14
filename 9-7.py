import sys
sys.setrecursionlimit(10**7)

n = 0
E = list([] for _ in range(102))
L = []
X = [-1] * 102
Y = []

def solution(edges, target):
    global n
    n = len(edges) + 1
    for p, c in edges:
        E[p - 1].append(c - 1)
    for i in range(n):
        E[i].sort()
        if len(E[i]) == 0:
            L.append(i)
        else:
            X[i] = 0

    for _ in range(10004):
        Y.append(dfs(0))

    answer = [-1]
    for k in range(10004):
        if is_ok(k, Y, target):
            answer = assign_stone(k, Y, target)
            break
    return answer

def assign_stone(k, Y, target):
    cnt = [0] * n
    for i in range(k):
        cnt[Y[i]] += 1

    answer = []
    for i in range(k):
        u = Y[i]
        cnt[u] -= 1
        if target[u] - 1 <= cnt[u]*3:
            answer.append(1)
            target[u] -= 1
        elif target[u] - 2 <= cnt[u]*3:
            answer.append(2)
            target[u] -= 2
        else:
            answer.append(3)
            target[u] -= 3
    return answer

def is_ok(k, Y, target):
    cnt = [0] * n
    for i in range(k):
        cnt[Y[i]] += 1
    for i in range(n):
        if target[i] < cnt[i] or cnt[i] * 3 < target[i]:
            return False
    return True

def dfs(u):
    if len(E[u]) == 0:
        return u
    ret = dfs(E[u][X[u]])
    X[u] = (X[u] + 1) % len(E[u])
    return ret