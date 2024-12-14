import sys
input=sys.stdin.readline

dd = [[-1, 0], [1, 0], [0, -1], [0, 1]]

N = 0; M = 0
A = [[0] * 21 for _ in range(21)]
B = [[0] * 21 for _ in range(21)]

def in_range(r, c):
	return 1 <= r <= N and 1 <= c <= N

def get_size(r, c, color, visited):
	global N, M, A, B
	visited[r][c] = 1

	ret=[0, 0]

	if A[r][c] == 0:
		ret[0] += 1
	else:
		ret[1] += 1

	for dr, dc in dd:
		nr = r + dr; nc = c + dc

		if in_range(nr, nc) == False:
			continue

		if visited[nr][nc] == 1:
			continue

		if A[nr][nc] == -1 or A[nr][nc] == -2:
			continue

		if A[nr][nc] != 0 and A[nr][nc] != color:
			continue

		v = get_size(nr, nc, color, visited)
		ret[0] += v[0]
		ret[1] += v[1]

	return ret

def fill_zero(r, c, color):
	global N, M, A, B

	A[r][c] = -2

	for dr, dc in dd:
		nr = r + dr; nc = c + dc

		if in_range(nr, nc) == False:
			continue

		if A[nr][nc] == -1 or A[nr][nc] == -2:
			continue

		if A[nr][nc] != 0 and A[nr][nc] != color:
			continue

		fill_zero(nr, nc, color)

def reset_visited(visited):
	global N, M, A, B
	for r in range(1, N + 1):
		for c in range(1, N + 1):
			if A[r][c] == 0:
				visited[r][c] = 0

def find_erase_bg():
	global N, M, A, B
	mx_r = -1; mx_c = -1
	mx = [ -1, -1 ]
	visited = [[0] * 21 for _ in range(21)]
	for r in range(1, N + 1):
		for c in range(1, N + 1):
			if A[r][c] == -1 or A[r][c] == -2 or A[r][c] == 0 or visited[r][c] == 1:
				continue

			reset_visited(visited)

			ret = get_size(r, c, A[r][c], visited)

			if ret[0] + ret[1] < 2:
				continue

			if mx[0] + mx[1] < ret[0] + ret[1] or (mx[0] + mx[1] == ret[0] + ret[1] and mx[0] <= ret[0]):
				mx_r = r; mx_c = c; mx[0] = ret[0]; mx[1] = ret[1]

	if mx_r == -1:
	   return 0

	fill_zero(mx_r, mx_c, A[mx_r][mx_c])

	return (mx[0] + mx[1]) * (mx[0] + mx[1])

# 3. 격자에 중력이 작용한다.
def apply_gravity():
	global N, M, A, B
	# 열 단위로 중력이 작용한다.
	for c in range(1, N + 1):
		# 아랫쪽 격자부터 중력이 작용한다.
		for r in range(N - 1, 0, -1):
			# 검은색 블럭이나 빈 블럭은 중력이 작용안한다.
			if A[r][c] == -1 or A[r][c] == -2:
				continue

			# A[r][c] 블럭이 떨어질 위치 (x, c)를 찾는다.
			# A[r][c] 와 연결된 연속된 빈 공간의 끝부분에 떨어진다.
			x = r + 1
			while x <= N:
				if A[x][c] != -2:
					break
				x += 1
			x -= 1

			# (x, c)가 존재하는 경우
			if r != x:
				# (r, c)에 있는 블록이 (x, c)에 떨어진다.
				A[x][c] = A[r][c]

				# (r, c)는 빈 블록이 된다.
				A[r][c] = -2

# 4. 격자가 90도 반시계 방향으로 회전한다.
def rotate_blocks():
	global N, M, A, B
	for r in range(1, N + 1):
		for c in range(1, N + 1):
			B[N - c + 1][r] = A[r][c]

	for r in range(1, N + 1):
		for c in range(1, N + 1):
			A[r][c] = B[r][c]

# 입력
N, M = map(int, input().split())
for r in range(1, N + 1):
	x = list(map(int, input().split()))
	for c in range(N):
		A[r][c + 1] = x[c]

ans = 0
while True:
	# 1. 크기가 가장 큰 블록 그룹을 찾는다
	# 2. 1에서 찾은 블록 그룹의 모든 블록을 제거한다.
	ret = find_erase_bg()

	# 더 이상 블록 그룹이 없는 경우 종료한다.
	if ret == 0:
		break

	# 획득한 점수를 누적한다.
	ans += ret

	# 3. 격자에 중력이 작용한다.
	apply_gravity()

	# 4. 격자가 90도 반시계 방향으로 회전한다.
	rotate_blocks()

	# 3. 격자에 중력이 작용한다.
	apply_gravity()

print(ans)