board = [['' for _ in range(51)] for _ in range(51)]

P = [[[-1, -1] for _ in range(51)] for _ in range(51)]
for i in range(1, 51):
    for j in range(1, 51):
        P[i][j] = [i, j]

def do_find(x):
    if P[x[0]][x[1]] == x:
        return x
    P[x[0]][x[1]] = do_find(P[x[0]][x[1]])
    return P[x[0]][x[1]]

def do_merge(x, y):
    px = do_find(x)
    py = do_find(y)
    P[py[0]][py[1]] = px

def solution(commands):
    answer = []
    for c in commands:
        cmd = c.split()

        if cmd[0] == 'UPDATE' and len(cmd) == 4:
            x = [int(cmd[1]), int(cmd[2])]
            px = do_find(x)
            board[px[0]][px[1]] = cmd[3]

        elif cmd[0] == 'UPDATE' and len(cmd) == 3:
            for r in range(1, 51):
                for c in range(1, 51):
                    if board[r][c] == cmd[1]:
                        board[r][c] = cmd[2]

        elif cmd[0] == 'MERGE':
            x = [int(cmd[1]), int(cmd[2])]
            y = [int(cmd[3]), int(cmd[4])]

            if x == y:
                continue

            px = do_find(x)
            py = do_find(y)

            value = ''
            if board[px[0]][px[1]] == '':
                value = board[py[0]][py[1]]
            else:
                value = board[px[0]][px[1]]

            board[px[0]][px[1]] = ''
            board[py[0]][py[1]] = ''

            do_merge(px, py)

            board[px[0]][px[1]] = value

        elif cmd[0] == 'UNMERGE':
            x = [int(cmd[1]), int(cmd[2])]
            px = do_find(x)
            ss = board[px[0]][px[1]]

            L = []
            for r in range(1, 51):
                for c in range(1, 51):
                    y = do_find([r, c])
                    if y == px:
                        L.append([r, c])

            for r, c in L:
                P[r][c] = [r, c]
                board[r][c] = ''

            board[x[0]][x[1]] = ss

        else:
            x = [int(cmd[1]), int(cmd[2])]
            x = do_find(x)
            if board[x[0]][x[1]] == '':
                answer.append('EMPTY')
            else:
                answer.append(board[x[0]][x[1]])

    return answer