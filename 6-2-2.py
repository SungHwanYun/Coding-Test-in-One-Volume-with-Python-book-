import heapq

INF = int(1e18)

def solution(N, M, edges, X, Z, Y):
    E = list([] for _ in range(N + 1))
    for u, v, d in edges:
        E[u].append([v, d])
        E[v].append([u, d])

    dist_X = [INF] * (N + 1)
    dijkstra(N, E, X, dist_X)

    dist_Z = [INF] * (N + 1)
    dijkstra(N, E, Z, dist_Z)

    answer = INF
    for y in Y:
        if dist_X[y]<INF and dist_Z[y]<INF and answer>dist_X[y]+dist_Z[y]:
            answer=dist_X[y]+dist_Z[y]

    if answer==INF: answer=-1
    return answer

def dijkstra(N, E, X, dist):
    selected = [False] * (N + 1)
    pq = []

    dist[X] = 0
    heapq.heappush(pq, [0, X])

    while len(pq) > 0:
        cost, here = heapq.heappop(pq)
        if dist[here] < cost or selected[here] == True:
            continue

        selected[here] = True

        for there, c in E[here]:
            next_dist = cost + c
            
            if selected[there] == False and dist[there] > next_dist:
                dist[there] = next_dist
                heapq.heappush(pq, [next_dist, there])

N, M = map(int, input().split())
edges = list(list(map(int, input().split())) for _ in range(M))
X, Z = map(int, input().split())
P=int(input())
Y=list(map(int, input().split()))
print(solution(N, M, edges, X, Z, Y))