import sys
input=sys.stdin.readline

dd = [ [0, 1], [1, 0], [0, -1], [-1, 0] ]

A = [[0] * 24 for _ in range(24)]
B = [[0] * 24 for _ in range(24)]
C = [[0] * 6 for _ in range(6)]
N = 0; M = 0; K = 0
visited = [[0] * 24 for _ in range(24)]

def in_range(r, c):
	return 1 <= r <= N and 1 <= c <= M

def dfs(r, c):
	global visited

	ret = 1
	visited[r][c] = 1

	for dr, dc in dd:
		nr = r + dr; nc = c + dc
		if in_range(nr, nc) == True and visited[nr][nc] == 0 and A[r][c] == A[nr][nc]:
			ret += dfs(nr, nc)

	return ret

def get_next_pos(x):
	r, c, d = x[0], x[1], x[2]

	nr = r + dd[d][0]; nc = c + dd[d][1]

	if in_range(nr, nc) == False:
		d = (d + 2) % 4
		nr = r + dd[d][0]
		nc = c + dd[d][1]

	x[:] = [nr, nc, d]

def rotate_cube(d):
	if d == 0:
		t = C[4][2]
		C[4][2] = C[2][3]
		C[2][3] = C[2][2]
		C[2][2] = C[2][1]
		C[2][1] = t
	elif d == 2:
		t = C[4][2]
		C[4][2] = C[2][1]
		C[2][1] = C[2][2]
		C[2][2] = C[2][3]
		C[2][3] = t
	elif d == 1:
		t = C[4][2]
		C[4][2] = C[3][2]
		C[3][2] = C[2][2]
		C[2][2] = C[1][2]
		C[1][2] = t
	else:
		t = C[4][2]
		C[4][2] = C[1][2]
		C[1][2] = C[2][2]
		C[2][2] = C[3][2]
		C[3][2] = t

def update_dir(r, c, d):
	a = C[4][2]; b = A[r][c]

	if a > b:
		d = (d + 1) % 4
	elif a < b:
		d = (d - 1 + 4) % 4

	return d

N, M, K = map(int, input().split())
for i in range(1, N + 1):
	x = list(map(int, input().split()))
	A[i][1:] = x[:]

for i in range(1, N + 1):
	for j in range(1, M + 1):
		for r in range(1, N + 1):
			for c in range(1, M + 1):
				visited[r][c] = 0
		B[i][j] = dfs(i, j) * A[i][j]

C[1][2] = 2
C[2][1] = 4; C[2][2] = 1; C[2][3] = 3
C[3][2] = 5
C[4][2] = 6

ans = 0; x = [1, 1, 0]
for _ in range(K):
	get_next_pos(x)
	rotate_cube(x[2])
    
	ans += B[x[0]][x[1]]

	x[2] = update_dir(x[0], x[1], x[2]);

print(ans)