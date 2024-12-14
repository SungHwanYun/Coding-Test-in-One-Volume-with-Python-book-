def in_range(board, r, c):
    return 0 <= r and r < len(board) and 0 <= c and c < len(board[0])

def solve(board, r1, c1, r2, c2):
    dd = [[-1, 0], [1, 0], [0, -1], [0, 1]]
    nxt=[]
    for dr, dc in dd:
        nr, nc = r1 + dr, c1 + dc

        if in_range(nr, nc) == False:
            continue

        if board[nr][nc] == 0:
            continue

        board[r1][c1] = 0
        ret = solve(board, r2, c2, nr, nc)

        board[r1][c1] = 1

        nxt.append(ret)

    if len(nxt) == 0:
        return 0

    if r1 == r2 and c1 == c2:
        return 1

    nxt.sort()

    if nxt[0] > 0:
        return -(nxt[-1] + 1)

    ret = None
    for i in range(len(nxt)):
        if nxt[i] <= 0:
            ret = nxt[i]
    return -ret + 1

def solution(board, aloc, bloc):
    answer = solve(board, aloc[0], aloc[1], bloc[0], bloc[1])
    return abs(answer)