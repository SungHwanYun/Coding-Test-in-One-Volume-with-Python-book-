import sys
sys.setrecursionlimit(10 ** 7)

D = list(list([0] * 2600 for i in range(54)) for j in range(54))
dd = [[1, 0], [0, -1], [0, 1], [-1, 0]]
dir_str = 'dlru'

def solution(n, m, x, y, r, c, k):
    dist = abs(r - x) + (c - y)
    if dist > k or ((dist & 0x1) is not (k & 0x1)):
        return 'impossible'

    x -= 1
    y -= 1
    r -= 1
    c -= 1

    make_D(r, c, 0, k, D, n, m)

    answer = ''
    while k > 0:
        idx = 0
        for dx, dy in dd:
            nx, ny = x + dx, y + dy
            if in_range(nx, ny, n, m) and D[nx][ny][k - 1] == 1:
                answer = answer + dir_str[idx]
                x, y = nx, ny
                break
            idx += 1
        if idx == 4:
            return 'impossible'
        k -= 1

    return answer

def in_range(r, c, n, m):
    return 0 <= r < n and 0 <= c < m

def make_D(r, c, kk, k, D, n, m):
    D[r][c][kk] = 1

    if kk == k:
        return

    for dr, dc in dd:
        nr, nc = r + dr, c + dc
        if in_range(nr, nc, n, m) and D[nr][nc][kk + 1] == 0:
            make_D(nr, nc, kk + 1, k, D, n, m)