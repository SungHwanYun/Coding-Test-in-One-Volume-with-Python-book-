import sys
import copy
input=sys.stdin.readline

dd = [[0, 0], [0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1]]

N , M = 0, 0
A = [[] for _ in range(50)]
B = []
C = [[0] * 50 for _ in range(50)]

def in_range(r, c):
	return 0 <= r < N and 0 <= c < N

def move_cloud(d, s):
	global B
	V = []
	for r, c in B:
		nr = (r + dd[d][0] * s) % N
		nc = (c + dd[d][1] * s) % N
		if nr < 0:
			nr += N
		if nc < 0:
			nc += N
		V.append([nr, nc])

	B.clear()
	B = copy.deepcopy(V)

def add_one_water():
	global A, B, C

	for r, c in B:
		A[r][c] += 1    

	for r in range(N):
		for c in range(N):
			C[r][c] = 0
	for r, c in B:
		C[r][c] = 1

	for r, c in B:
		d = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
		for dr, dc in d:
			nr = r + dr; nc = c + dc
			if in_range(nr, nc) == True and A[nr][nc] > 0:
				A[r][c] += 1

	B.clear()

def build_cloud():
	global A, B, C
	for r in range(N):
		for c in range(N):
			if A[r][c] >= 2 and C[r][c] == 0:
				B.append([r, c])
				A[r][c] -= 2

N, M = map(int, input().split())
for r in range(N):
	A[r] = list(map(int, input().split()))

B.append([N - 1, 0]); B.append([N - 1, 1])
B.append([N - 2, 0]); B.append([N - 2, 1])

for _ in range(M):
	d, s = map(int, input().split())

	move_cloud(d, s)

	add_one_water()

	build_cloud()

ans = 0
for r in range(N):
	for c in range(N):
		ans += A[r][c]
print(ans)