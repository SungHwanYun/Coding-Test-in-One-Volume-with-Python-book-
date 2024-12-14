from queue import Queue
import copy
from itertools import permutations

def in_range(r, c):
    return 0 <= r < 4 and 0 <= c < 4

def get_move_count(board, src, dst):
    dd = [[0, -1], [0, 1], [-1, 0], [1, 0]]
    visited = [[0] * 4 for _ in range(4)]
    dist = [[0] * 4 for _ in range(4)]

    Q = Queue()

    Q.put(src)
    dist[src[0]][src[1]] = 0
    visited[src[0]][src[1]] = 1

    while Q.empty() == False:
        r, c = Q.get()

        if r == dst[0] and c == dst[1]:
            return dist[r][c] + 1

        for dr, dc in dd:
            nr = r + dr; nc = c + dc
            if in_range(nr, nc) == True and visited[nr][nc] == 0:
                Q.put([nr, nc])
                dist[nr][nc] = dist[r][c] + 1
                visited[nr][nc] = 1

        for dr, dc in dd:
            nr = r; nc = c
            while True:
                if in_range(nr + dr, nc + dc) == False:
                    break

                nr += dr; nc += dc
                if board[nr][nc] != 0:
                    break

            if visited[nr][nc] == 0:
                Q.put([nr, nc])
                dist[nr][nc] = dist[r][c] + 1
                visited[nr][nc] = 1

    return int(1e8)

def solution(board, r, c):
    X = [[] for i in range(7)]
    arr = []
    for i in range(4):
        for j in range(4):
            if board[i][j] == 0:
                continue

            if len(X[board[i][j]]) == 0:
                arr.append(board[i][j])
                
            X[board[i][j]].append((i, j))

    n = len(arr)
    
    ans = int(1e9)
    for p in permutations(arr):
        _board = copy.deepcopy(board)

        d = [[0]*2 for _ in range(n)]

        d[0][0] = get_move_count(_board, (r, c), X[p[0]][0]) + get_move_count(_board, X[p[0]][0], X[p[0]][1])
        d[0][1] = get_move_count(_board, (r, c), X[p[0]][1]) + get_move_count(_board, X[p[0]][1], X[p[0]][0])
        _board[X[p[0]][0][0]][X[p[0]][0][1]] = _board[X[p[0]][1][0]][X[p[0]][1][1]] = 0

        for i in range(1, n):
            d[i][0] = min(d[i-1][0] + get_move_count(_board, X[p[i-1]][1], X[p[i]][0]), \
                d[i-1][1] + get_move_count(_board, X[p[i-1]][0], X[p[i]][0])) + \
                get_move_count(_board, X[p[i]][0], X[p[i]][1])

            d[i][1] = min(d[i-1][0] + get_move_count(_board, X[p[i-1]][1], X[p[i]][1]), \
                d[i-1][1] + get_move_count(_board, X[p[i-1]][0], X[p[i]][1])) + \
                get_move_count(_board, X[p[i]][1], X[p[i]][0])

            _board[X[p[i]][0][0]][X[p[i]][0][1]] = _board[X[p[i]][1][0]][X[p[i]][1][1]] = 0

        ans = min(ans, d[n-1][0], d[n-1][1])

    return ans