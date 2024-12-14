import sys
input=sys.stdin.readline

dd = [[-1, 0], [1, 0], [0, -1], [0, 1]]
point = [0, 1, 10, 100, 1000]

N = 0
A = [[0] * 21 for _ in range(21)]
B = [[] for _ in range(401)]

def in_range(r, c):
	return 1 <= r <= N and 1 <= c <= N

def set_student(x, y):
	mx_r = 1; mx_c = 1; mx_empty = -1; mx_favorate = -1;

	for r in range(1, N + 1):
		for c in range(1, N + 1):
			if A[r][c] != 0:
				continue

			empty = 0
			favorate = 0
			for dr, dc in dd:
				nr = r + dr; nc = c + dc
				if in_range(nr, nc) == False:
					continue
				if A[nr][nc] == 0:
					empty += 1
				elif A[nr][nc] in y:
					favorate += 1

			if mx_favorate < favorate or (mx_favorate == favorate and mx_empty < empty):
				mx_r = r; mx_c = c; mx_empty = empty; mx_favorate = favorate

	A[mx_r][mx_c] = x

N = int(input())
for _ in range(N*N):
	b = list(map(int, input().split()))
	B[b[0]] = b[1:]
	
	set_student(b[0], b[1:])

ans = 0
for r in range(1, N + 1, 1):
    for c in range(1, N + 1, 1):
        favorate = 0
        for dr, dc in dd:
            nr = r + dr; nc = c + dc
            if in_range(nr, nc) == True and (A[nr][nc] in B[A[r][c]]):
                favorate += 1
        ans += point[favorate]
        
print(ans)