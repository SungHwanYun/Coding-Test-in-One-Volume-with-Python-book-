def update_board(board, r1, c1, r2, c2, degree):
    for r in range(r1, r2 + 1, 1):
        for c in range(c1, c2 + 1, 1):
            board[r][c] += degree

def solution(board, skill):
    for type, r1, c1, r2, c2, degree in skill:
        if type == 1:
            update_board(board, r1, c1, r2, c2, -degree)
        else:
            update_board(board, r1, c1, r2, c2, degree)

    answer = 0
    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] > 0:
               answer += 1
    return answer