import sys
input=sys.stdin.readline

def solution(board, aloc):
    dd = [[-1, 0], [1, 0], [0, -1], [0, 1]]
    
    for i in range(4):
        for j in range(4):
            for k in range(4):
                iloc=[aloc[0] + dd[i][0], aloc[1] + dd[i][1]]
                jloc=[iloc[0] + dd[j][0], iloc[1] + dd[j][1]]
                kloc=[jloc[0] + dd[k][0], jloc[1] + dd[k][1]]

                if get_apple(board, aloc, iloc, jloc, kloc) >= 2:
                    return 1

    return 0

def get_apple(board, aloc, iloc, jloc, kloc):
    apple_num = 0

    if in_range(iloc) == False or in_range(jloc) == False:
        return -1
    
    if board[iloc[0]][iloc[1]] == -1 or board[jloc[0]][jloc[1]] == -1:
        return -1
    
    if aloc == jloc:
        return -1

    apple_num = board[iloc[0]][iloc[1]] + board[jloc[0]][jloc[1]]
    
    if in_range(kloc) and board[kloc[0]][kloc[1]] == 1 and iloc != kloc:
        apple_num += 1

    return apple_num

def in_range(loc):
    return 0 <= loc[0] <= 4 and 0 <= loc[1] <= 4

board = []
for _ in range(5):
    board.append(list(map(int, input().split())))
aloc = list(map(int, input().split()))

print(solution(board, aloc))