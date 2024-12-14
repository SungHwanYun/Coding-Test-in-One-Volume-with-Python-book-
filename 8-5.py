def solution(board, aloc, bloc):
    return solve(board, aloc, bloc, 0)

def solve(board, aloc, bloc, apple_diff):
    if board[aloc[0]][aloc[1]] == -1 and board[bloc[0]][bloc[1]] == -1:
        if apple_diff > 0:
            return 1
        return 0

    remained_apple = 0
    for i in range(5):
        remained_apple += board[i].count(1)
    if remained_apple == 0:
        if apple_diff > 0:
            return 1
        return 0

    dd = [[-1, 0], [1, 0], [0, -1], [0, 1]]
    try_count = 0
    for dr, dc in dd:
        r, c = aloc[0] + dr, aloc[1] + dc
        if in_range([r, c]) and board[r][c] != -1 and [r, c] != bloc:
            try_count += 1
            prv_value = board[aloc[0]][aloc[1]]
            board[aloc[0]][aloc[1]] = -1
            ret = solve(board, bloc, [r, c], -(apple_diff + board[r][c]) + 1)

            board[aloc[0]][aloc[1]] = prv_value

            if ret == 0:
                return 1

    if try_count == 0:
        prv_value = board[aloc[0]][aloc[1]]
        board[aloc[0]][aloc[1]] = -1
        ret = solve(board, bloc, aloc, -apple_diff + 1)
        board[aloc[0]][aloc[1]] = prv_value
        if ret == 0:
            return 1

    return 0

def in_range(loc):
    return 0 <= loc[0] <= 4 and 0 <= loc[1] <= 4

board = []
for _ in range(5):
    board.append(list(map(int, input().split())))
loc = list(map(int, input().split()))
aloc, bloc = loc[:2], loc[2:]

print(solution(board, aloc, bloc))