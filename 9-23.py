from queue import Queue
import sys
input=sys.stdin.readline

RIGHT = 1; LEFT = 2; UP = 3; DOWN = 4
R_MASK = (1 << RIGHT)
L_MASK = (1 << LEFT)
U_MASK = (1 << UP)
D_MASK = (1 << DOWN)

dd = [[0, 0], [0, 1], [0, -1], [-1, 0], [1, 0]]

N = 0; M = 0; K = 0
A = [[0] * 24 for _ in range(24)]
B = [[0] * 24 for _ in range(24)]
C = [[0] * 24 for _ in range(24)]
W = [[0] * 24 for _ in range(24)]

def in_range(r, c):
	return 1 <= r <= N and 1 <= c <= M

def build_B_sub(sr, sc, d):
	global A, B, C, W, N, M, K
	visited = [[0] * 24 for _ in range(24)]

	Q = Queue()

	Q.put([sr + dd[d][0], sc + dd[d][1], 5])
	visited[sr + dd[d][0]][sc + dd[d][1]] = 1

	while Q.empty() == False:
		r, c, t = Q.get()

		B[r][c] += t

		if t == 1:
			continue

		if d == RIGHT:
			nr = r - 1; nc = c + 1
			if in_range(nr, nc) == True and visited[nr][nc] == 0 and \
				(W[r][c] & U_MASK) == 0 and (W[nr][nc] & L_MASK) == 0:
				Q.put([nr, nc, t - 1])
				visited[nr][nc] = 1

			nr = r
			if in_range(nr, nc) == True and visited[nr][nc] == 0 and \
                (W[r][c] & R_MASK) == 0:
				Q.put([nr, nc, t - 1])
				visited[nr][nc] = 1

			nr = r + 1
			if in_range(nr, nc) == True and visited[nr][nc] == 0 and \
				(W[r][c] & D_MASK) == 0 and (W[nr][nc] & L_MASK) == 0:
				Q.put([nr, nc, t - 1])
				visited[nr][nc] = 1
		elif d == LEFT:
			nr = r - 1; nc = c - 1
			if in_range(nr, nc) == True and visited[nr][nc] == 0 and \
				(W[r][c] & U_MASK) == 0 and (W[nr][nc] & R_MASK) == 0:
				Q.put([nr, nc, t - 1])
				visited[nr][nc] = 1

			nr = r
			if in_range(nr, nc) == True and visited[nr][nc] == 0 and \
                (W[r][c] & L_MASK) == 0:
				Q.put([nr, nc, t - 1])
				visited[nr][nc] = 1

			nr = r + 1
			if in_range(nr, nc) == True and visited[nr][nc] == 0 and \
				(W[r][c] & D_MASK) == 0 and (W[nr][nc] & R_MASK) == 0:
				Q.put([nr, nc, t - 1])
				visited[nr][nc] = 1
		elif d == UP:
			nr = r - 1; nc = c - 1
			if in_range(nr, nc) == True and visited[nr][nc] == 0 and \
				(W[r][c] & L_MASK) == 0 and (W[nr][nc] & D_MASK) == 0:
				Q.put([nr, nc, t - 1])
				visited[nr][nc] = 1

			nc = c
			if in_range(nr, nc) == True and visited[nr][nc] == 0 and \
                (W[r][c] & U_MASK) == 0:
				Q.put([nr, nc, t - 1])
				visited[nr][nc] = 1

			nc = c + 1
			if in_range(nr, nc) == True and visited[nr][nc] == 0 and \
				(W[r][c] & R_MASK) == 0 and (W[nr][nc] & D_MASK) == 0:
				Q.put([nr, nc, t - 1])
				visited[nr][nc] = 1
		else:
			nr = r + 1; nc = c - 1
			if in_range(nr, nc) == True and visited[nr][nc] == 0 and \
				(W[r][c] & L_MASK) == 0 and (W[nr][nc] & U_MASK) == 0:
				Q.put([nr, nc, t - 1])
				visited[nr][nc] = 1

			nc = c
			if in_range(nr, nc) == True and visited[nr][nc] == 0 and \
                (W[r][c] & D_MASK) == 0:
				Q.put([nr, nc, t - 1])
				visited[nr][nc] = 1

			nc = c + 1
			if in_range(nr, nc) == True and visited[nr][nc] == 0 and \
				(W[r][c] & R_MASK) == 0 and (W[nr][nc] & U_MASK) == 0:
				Q.put([nr, nc, t - 1])
				visited[nr][nc] = 1

def build_B():
	global A, B, C, W, N, M, K
	for r in range(1, N + 1):
		for c in range(1, M + 1):
			if 0 < C[r][c] < 5:
				build_B_sub(r, c, C[r][c])

N, M, K = map(int, input().split())
for r in range(1, N + 1):
	x = list(map(int, input().split()))
	C[r][1:] = x[:]

w = int(input())
for _ in range(w):
	r, c, t = map(int, input().split())

	if t == 0:
		W[r][c] |= U_MASK
		W[r - 1][c] |= D_MASK
	else:
		W[r][c] |= R_MASK
		W[r][c + 1] |= L_MASK

build_B()

for step in range(1, 101, 1):
	for r in range(1, N + 1):
		for c in range(1, M + 1):
			A[r][c] += B[r][c]

	X = [[0] * 24 for _ in range(24)]
	for r in range(1, N + 1):
		for c in range(1, M + 1):
			nr = r; nc = c + 1
			if in_range(nr, nc) == True and (W[r][c] & R_MASK) == 0:
				if A[r][c] - A[nr][nc] > 0:
					diff = (A[r][c] - A[nr][nc]) // 4
				else:
					diff = -((A[nr][nc] - A[r][c]) // 4)
				X[r][c] -= diff
				X[nr][nc] += diff

			nr = r + 1; nc = c
			if in_range(nr, nc) == True and (W[r][c] & D_MASK) == 0:
				if A[r][c] - A[nr][nc] > 0:
					diff = (A[r][c] - A[nr][nc]) // 4
				else:
					diff = -((A[nr][nc] - A[r][c]) // 4)
				X[r][c] -= diff
				X[nr][nc] += diff

	for r in range(1, N + 1):
		for c in range(1, M + 1):
			A[r][c] += X[r][c]

	for c in range(1, M + 1):
		if A[1][c] >= 1: A[1][c] -= 1
		if A[N][c] >= 1: A[N][c] -= 1
	for r in range(2, N):
		if A[r][1] >= 1: A[r][1] -= 1
		if A[r][M] >= 1: A[r][M] -= 1

	is_ok = 1
	for r in range(1, N + 1):
		for c in range(1, M + 1):
			if C[r][c] == 5 and A[r][c] < K:
				is_ok = 0
	if is_ok:
		print(step)
		sys.exit()

print(101)