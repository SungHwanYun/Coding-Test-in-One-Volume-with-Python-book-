import sys
input=sys.stdin.readline

from collections import deque

def solution(board, sr, sc):
    tr, tc = 0, 0
    for r in range(5):
        for c in range(5):
            if board[r][c] == 1:
                tr, tc = r, c

    return minimum_move(board, sr, sc, tr, tc)

def minimum_move(board, sr, sc, tr, tc):
    dd = [[-1, 0], [1, 0], [0, -1], [0, 1]]
    visited = [[0] * 5 for _ in range(5)]
    dist = [[0] * 5 for _ in range(5)]

    q = deque()
    q.append([sr, sc])
    
    visited[sr][sc] = 1
    dist[sr][sc] = 0
    
    while len(q) != 0:
        r, c = q.popleft()

        if r == tr and c == tc:
            return dist[r][c]

        for dr, dc in dd:
            nr = r + dr; nc = c + dc
            if in_range(nr, nc) == True and visited[nr][nc] == 0 and\
                board[nr][nc] != -1:
                q.append([nr, nc])
                dist[nr][nc] = dist[r][c] + 1
                visited[nr][nc] = 1

    return -1

def in_range(r, c):
    return 0 <= r <= 4 and 0 <= c <= 4

board = list(list(map(int, input().split())) for _ in range(5))
sr, sc = map(int, input().split())
print(solution(board, sr, sc))