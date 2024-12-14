import sys
input=sys.stdin.readline

def solution(board, aloc):
    return solve(board, aloc, 3)

def solve(board, aloc, apple_num):
    if apple_num == 0:
        return 0
    
    ret = -1
    
    dd = [[-1, 0], [1, 0], [0, -1], [0, 1]]
    for dr, dc in dd:
        r, c = aloc[0] + dr, aloc[1] + dc
        if in_range([r, c]) and board[r][c] != -1:
            prv_value = board[aloc[0]][aloc[1]]
            board[aloc[0]][aloc[1]] = -1
            
            cur_ret = solve(board, [r, c], apple_num - board[r][c])
            if cur_ret != -1:
                cur_ret += 1
            
            if cur_ret != -1:
                if ret == -1 or cur_ret < ret:
                    ret = cur_ret
            
            board[aloc[0]][aloc[1]] = prv_value
         
    return ret

def in_range(loc):
    return 0 <= loc[0] <= 4 and 0 <= loc[1] <= 4

board = []
for _ in range(5):
    board.append(list(map(int, input().split())))
aloc = list(map(int, input().split()))

print(solution(board, aloc))