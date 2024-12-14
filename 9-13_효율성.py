def update_board(board, r1, c1, r2, c2, degree):
    board[r1][c1] += degree
    if c2 + 1 < len(board[0]):
        board[r1][c2 + 1] -= degree
    if r2 + 1 < len(board):
        board[r2 + 1][c1] -= degree
    if r2 + 1 < len(board) and c2 + 1 < len(board[0]):
        board[r2 + 1][c2 + 1] += degree

def solution(board, skill):
    board_diff=[[0] * len(board[0]) for _ in range(len(board))]
    for type, r1, c1, r2, c2, degree in skill:
        if type == 1:
            update_board(board_diff, r1, c1, r2, c2, -degree)
        else:
            update_board(board_diff, r1, c1, r2, c2, degree)

    for r in range(len(board)):
        for c in range(1, len(board[0]), 1):
            board_diff[r][c] += board_diff[r][c - 1]

    for c in range(len(board[0])):
        for r in range(1, len(board), 1):
            board_diff[r][c] += board_diff[r - 1][c]

    answer = 0
    for r in range(len(board)):
        for c in range(len(board[0])):
            if board[r][c] + board_diff[r][c] > 0:
                answer+=1
    return answer