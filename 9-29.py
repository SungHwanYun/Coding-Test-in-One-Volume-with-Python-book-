import sys
input=sys.stdin.readline

N, M = 0, 0
A = [[] for _ in range(50)]
B = [[0] * 50 for _ in range(50)]
C = [0] * 2500
D = [0] * 2500
ans = [0] * 4

def in_range(r, c):
	return 0 <= r < N and 0 <= c < N

def build_BC():
	global A, B, C

	dd = [[0, -1], [1, 0], [0, 1], [-1, 0]]

	r = c = N // 2; d = 0; cnt = 0
	remain_move = total_move = 1
	step = 1

	while True:
		if in_range(r, c) == False:
			break

		B[r][c] = cnt
		C[cnt] = A[r][c]
		cnt += 1

		r = r + dd[d][0]; c = c + dd[d][1]
		remain_move -= 1

		if remain_move == 0 and step == 2:
			total_move += 1
			remain_move = total_move
			step = 1
			d = (d + 1) % 4
		elif remain_move == 0:
			remain_move = total_move
			step = 2
			d = (d + 1) % 4

def move_thing_slow():
	global C

	empty = 0

	for i in range(1, N * N):
		if C[i] == 0:
			empty += 1
			continue

		if empty == 0:
			continue

		for j in range(1, i):
			if C[j] == 0:
				break
		C[j] = C[i]
		C[i] = 0

def move_thing_fast():
	global C

	i = 1; j = 1
	while i < N * N:
		if C[i] != 0:
			if i != j: 
				C[j] = C[i]
				C[i] = 0

			j += 1

		i += 1

def explode_thing_sub():
	global C, ans

	i = 1; j = 0; ret = 0
	while i < N * N:
		if C[i] == 0:
			i += 1
			continue

		for j in range(i + 1, N * N):
			if C[i] != C[j]:
				break

		if j - i >= 4:
			ans[C[i]] += j - i

			for k in range(i, j):
				C[k] = 0

			ret = 1

		i = j

	return ret

def explode_thing():
	while True:
		if explode_thing_sub() == 0:
			break

		move_thing_slow()

def change_thing():
	global C, D

	i = 1; j = 0; k = 1
	while i < N * N:
		if C[i] == 0: 
			i += 1
			continue

		if k >= N * N - 1:
			break;

		for j in range(i + 1, N * N):
			if C[i] != C[j]:
				break

		D[k] = j - i; k += 1
		D[k] = C[i]; k += 1

		i = j

	for i in range(k):
		C[i] = D[i]
	for i in range(k, N * N):
		C[i] = 0

N, M = map(int, input().split())
for r in range(N):
	A[r] = list(map(int, input().split()))

build_BC()

for _ in range(M):
	d, s = map(int, input().split())

	dd = [[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]]

	r = c = N // 2
	for _ in range(s):
		r = r + dd[d][0]; c = c + dd[d][1]
		C[B[r][c]] = 0

	move_thing_slow()

	explode_thing()

	change_thing()

print(ans[1] + 2 * ans[2] + 3 * ans[3])