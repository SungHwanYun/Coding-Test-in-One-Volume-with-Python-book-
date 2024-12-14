import sys
input=sys.stdin.readline

from queue import PriorityQueue

INF = int(1e18)

def solution(N, M, edges, X, Z):
    E = list([] for _ in range(N + 1))
    for p, c, d in edges:
        E[p].append([c, d])
    return dijkstra(N, E, X, Z)

def dijkstra(N, E, X, Z):
    selected = [False] * (N + 1)
    dist = [INF] * (N + 1)
    pq = PriorityQueue()
    
    dist[X] = 0
    pq.put([0, X])

    while pq.empty() == False:
        cost, here = pq.get()
    
        if dist[here] < cost or selected[here] == True: continue

        selected[here] = True
        
        if here == Z:
            return dist[here]
        
        for there, c in E[here]:
            next_dist = cost + c

            if selected[there] == False and dist[there] > next_dist:
                dist[there] = next_dist
                pq.put([next_dist, there])

    return -1

N, M = map(int, input().split())
edges = list(list(map(int, input().split())) for _ in range(M))
X, Z = map(int, input().split())
print(solution(N, M, edges, X, Z))