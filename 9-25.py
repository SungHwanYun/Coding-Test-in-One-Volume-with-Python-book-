import sys
input=sys.stdin.readline

N = 0; K = 0
A = [[0] * 100 for _ in range(100)]
B = [[0] * 100 for _ in range(100)]
S = [0] * 100

def is_ok():
	global N, K, S

	mx, mn = 0, int(1e9)
	for i in range(N):
		for j in range(S[i]):
			mx = max(mx, A[i][j])
			mn = min(mn, A[i][j])
	return mx - mn <= K

def add_one_fish():
	global N, A

	mn = int(1e9)
	for i in range(N):
		mn = min(mn, A[i][0])

	for i in range(N):
		if A[i][0] == mn:
			A[i][0] += 1

def move_column(src, dst):
	global A, S
	for i in range(S[src]):
		A[dst][i] = A[src][i]
	S[dst] = S[src]

def build_up_sub(e):
	global N, K, A, B, S
	for i in range(e, -1, -1):
		j = e + 1; k = 0
		while k < S[i]:
			A[j][S[j]] = A[i][k]
			S[j] += 1
			j += 1
			k += 1

	i = 0; j = e + 1
	while j < N:
		move_column(j, i)
		i += 1; j += 1

	N = N - e - 1

def build_up():
	global N, S
	while True:
		i = 1
		while i < N:
			if S[i] == 1: break
			i += 1

		if S[0] > N - i: break
		build_up_sub(i - 1)

def adjust_fish_sub(r1, c1, r2, c2):
	global A, B
	if A[r1][c1] > A[r2][c2]:
		diff = (A[r1][c1] - A[r2][c2]) // 5
		B[r1][c1] -= diff
		B[r2][c2] += diff
	else:
		diff = (A[r2][c2] - A[r1][c1]) // 5
		B[r1][c1] += diff
		B[r2][c2] -= diff

def adjust_fish():
	global N, A, B, S
    
	for r in range(N):
		for c in range(S[r]):
			B[r][c] = 0

	for r in range(N):
		for c in range(S[r]):
			if r < N - 1 and c < S[r + 1]:
				adjust_fish_sub(r, c, r + 1, c)

			if c + 1 < S[r]:
				adjust_fish_sub(r, c, r, c + 1)

	for r in range(N):
		for c in range(S[r]):
			A[r][c] += B[r][c]

def spread_fish():
	global N, K, A, B, S
	k = 0
	for i in range(N):
		for j in range(S[i]):
			B[k][0] = A[i][j]
			k += 1

	N = k
	for i in range(N):
		A[i][0] = B[i][0]
		S[i] = 1

def build_up_half():
	global N, K, A, B, S
    
	e = (N // 2) - 1
	for c in range(S[0] - 1, -1, -1):
		r1 = e; r2 = e + 1
		while r1 >= 0:
			A[r2][S[r2]] = A[r1][c]
			S[r2] += 1
			r1 -= 1; r2 += 1

	r1 = 0; r2 = e + 1
	while r2 < N:
		move_column(r2, r1)
		r1 += 1; r2 += 1
	
	N = N // 2

N, K = map(int, input().split())
x = list(map(int, input().split()))
for i in range(N):
	A[i][0] = x[i]
	S[i] = 1

step = 0
while True:
	if is_ok() == True: break

	add_one_fish()

	build_up()

	adjust_fish()

	spread_fish()

	build_up_half()
	build_up_half()

	adjust_fish()

	spread_fish()
	step += 1

print(step)