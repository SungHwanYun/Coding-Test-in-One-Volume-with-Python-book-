import sys
input=sys.stdin.readline

LEFT = 1; LEFT_UP = 2; UP = 3; RIGHT_UP = 4; RIGHT = 5; RIGHT_DOWN = 6; DOWN = 7; LEFT_DOWN = 8

dd=[ [0, 0], [0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1] ]

N = 4; Sr = 0; Sc = 0

A = [ [[0] * 10 for _ in range(5)] for _ in range(5)]
C = [ [[0] * 104 for _ in range(5)] for _ in range(5)]
X = [ [[0] * 10 for _ in range(5)] for _ in range(5)]

def in_range(r, c):
	return 1 <= r <= N and 1 <= c <= N

def copy_A():
	global A, X, N;
	for r in range(1, N + 1):
		for c in range(1, N + 1):
			for d in range(9):
				X[r][c][d] = A[r][c][d]

def paste_A():
	global A, X, N;
	for r in range(1, N + 1):
		for c in range(1, N + 1):
			for d in range(9):
				A[r][c][d] += X[r][c][d]

def is_smell(step, r, c):
	if step == 1: return 0

	if step == 2: return C[r][c][1]

	return C[r][c][step - 1] | C[r][c][step - 2]

def move_fish(step):
	global A, C, X, N, Sr, Sc;

	T = [ [[0] * 10 for _ in range(5)] for _ in range(5)]

	for r in range(1, N + 1):
		for c in range(1, N + 1):
			for d in range(1, 9):
				if A[r][c][d] == 0: continue

				nr, nc, nd, i = 0, 0, 0, 0
				while i < 8:
					nd = d - i;
					if nd <= 0: nd += 8
					nr = r + dd[nd][0]; nc = c + dd[nd][1]
					if in_range(nr, nc) == True and not(nr == Sr and nc == Sc) and \
                        is_smell(step, nr, nc) == 0:
						break
					i += 1

				if i == 8:
					nr, nc, nd = r, c, d

				T[nr][nc][nd] += A[r][c][d]
				T[nr][nc][0] += A[r][c][d]

	for r in range(1, N + 1):
		for c in range(1, N + 1):
			for d in range(9):
				A[r][c][d] = T[r][c][d]

def reset_A(r, c):
	global A;
	for d in range(9):
		A[r][c][d] = 0

xx = [ [-1, 0], [0, -1], [1, 0], [0, 1] ]

def get_sum(d1, d2, d3):
	visited = [[0] * 5 for _ in range(5)]
	visited[Sr + xx[d1][0]][Sc + xx[d1][1]] = 1
	visited[Sr + xx[d1][0] + xx[d2][0]][Sc + xx[d1][1] + xx[d2][1]] = 1
	visited[Sr + xx[d1][0] + xx[d2][0] + xx[d3][0]][Sc + xx[d1][1] + xx[d2][1] + xx[d3][1]] = 1

	ret = 0
	for r in range(1, N + 1):
		for c in range(1, N + 1):
			if visited[r][c]:
				ret += A[r][c][0]
	return ret

def move_shark(step):
	global A, C, X, N, Sr, Sc;
	x, y, z, mx = 0, 0, 0, -1

	for i in range(4):
		r = Sr + xx[i][0]; c = Sc + xx[i][1]
		if in_range(r, c) == False: continue
		for j in range(4):
			r = Sr + xx[i][0] + xx[j][0]; c = Sc + xx[i][1] + xx[j][1]
			if in_range(r, c) == False: continue
			for k in range(4):
				r = Sr + xx[i][0] + xx[j][0] + xx[k][0]; c = Sc + xx[i][1] + xx[j][1] + xx[k][1]
				if in_range(r, c) == False: continue
				s = get_sum(i, j, k)
				if s > mx:
					mx, x, y, z = s, i, j, k

	r = Sr + xx[x][0]; c = Sc + xx[x][1]
	if A[r][c][0] > 0:
		reset_A(r, c)
		C[r][c][step] = 1

	r += xx[y][0]; c += xx[y][1]
	if A[r][c][0] > 0:
		reset_A(r, c)
		C[r][c][step] = 1

	r += xx[z][0]; c += xx[z][1]
	if A[r][c][0] > 0:
		reset_A(r, c)
		C[r][c][step] = 1

	Sr, Sc = r, c

M, S = map(int, input().split())
for _ in range(M):
	r, c, d = map(int, input().split())
	A[r][c][d] += 1
Sr, Sc = map(int, input().split())

for r in range(1, N + 1):
	for c in range(1, N + 1):
		for d in range(1, 9):
			A[r][c][0] += A[r][c][d]

for step in range(1, S + 1):
	copy_A()

	move_fish(step)

	move_shark(step)

	paste_A()

ans = 0
for r in range(1, N + 1):
	for c in range(1, N + 1):
		ans += A[r][c][0]
print(ans)