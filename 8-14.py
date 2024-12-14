import sys

input = sys.stdin.readline

from collections import deque
from itertools import permutations


def solution(board, sr, sc):
    source = list([] for _ in range(6))
    for r in range(5):
        for c in range(5):
            if board[r][c] > 0:
                source[board[r][c] - 1] = [r, c]

    answer = -1
    for target in permutations(source):
        ret = 0
        r, c = sr, sc
        for nr, nc in target:
            x = get_move_count(board, r, c, nr, nc)

            if x == -1:
                ret = -1
                break

            ret += x
            r, c = nr, nc

        if ret != -1:
            if answer == -1 or answer > ret:
                answer = ret

    return answer


def get_move_count(board, sr, sc, tr, tc):
    dd = [[0, -1], [0, 1], [-1, 0], [1, 0]]
    visited = [[0] * 5 for _ in range(5)]
    dist = [[0] * 5 for _ in range(5)]

    q = deque()
    q.append([sr, sc])
    visited[sr][sc] = 1

    while len(q) != 0:
        r, c = q.popleft()

        if r == tr and c == tc:
            return dist[r][c]

        for dr, dc in dd:
            nr = r + dr;
            nc = c + dc
            if in_range(nr, nc) == True and visited[nr][nc] == 0 and \
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