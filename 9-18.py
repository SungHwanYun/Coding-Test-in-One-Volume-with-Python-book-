INF = int(1e8)
E = [[INF]*204 for _ in range(204)]
D = [[0]*204 for _ in range(204)]

def solution(n, s, a, b, fares):
    for i in range(1, n + 1, 1):
        E[i][i] = 0

    for u, v, w in fares:
        E[u][v] = E[v][u] = w

    for i in range(1, n + 1, 1):
        for j in range(1, n + 1, 1):
            D[i][j] = E[i][j]
    for k in range(1, n + 1): # k: 거쳐가는 정점
        for i in range(1, n + 1): # i: 출발 정점
            for j in range(1, n + 1): # j: 도착 정점
                if D[i][k] + D[k][j] < D[i][j]:
                    D[i][j] = D[i][k] + D[k][j]

    answer = INF
    for k in range(1, n + 1, 1):
        ret = D[s][k] + D[k][a] + D[k][b]
        answer = min(answer, ret)
        
    return answer