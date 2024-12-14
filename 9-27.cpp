#include <bits/stdc++.h>
using namespace std;

int dr[4] = { 0,0,-1,1 }, dc[4] = { -1,1,0,0 };
int N, M, A[21][21], B[21][21], visited[21][21];

bool in_range(int r, int c) {
	return 1 <= r && r <= N && 1 <= c && c <= N;
}

vector<int> get_size(int r, int c, int color) {
	visited[r][c] = 1;
	vector<int> ret = { 0,0 };
	if (A[r][c] == 0) ret[0]++;
	else ret[1]++;
	for (int i = 0; i < 4; i++) {
		int nr = r + dr[i], nc = c + dc[i];
		if (in_range(nr, nc) == 0) continue;
		if (visited[nr][nc] == 1) continue;
		if (A[nr][nc] == -1 || A[nr][nc] == -2) continue;
		if (A[nr][nc] != 0 && A[nr][nc] != color) continue;
		vector<int> v = get_size(nr, nc, color);
		ret[0] += v[0];
		ret[1] += v[1];
	}
	return ret;
}

void fill_zero(int r, int c, int color) {
	A[r][c] = -2;
	for (int i = 0; i < 4; i++) {
		int nr = r + dr[i], nc = c + dc[i];
		if (in_range(nr, nc) == 0) continue;
		if (A[nr][nc] == -1 || A[nr][nc] == -2) continue;
		if (A[nr][nc] != 0 && A[nr][nc] != color) continue;
		fill_zero(nr, nc, color);
	}
}

void reset_visited() {
	for (int r = 1; r <= N; r++)
		for (int c = 1; c <= N; c++)
			if (A[r][c] == 0)
				visited[r][c] = 0;
}
int find_erase_bg() {
	int mx_r = -1, mx_c = -1, mx[2] = { -1,-1 };
	int r, c;
	for (r = 1; r <= N; r++)
		for (c = 1; c <= N; c++)
			visited[r][c] = 0;

	for (r = 1; r <= N; r++) {
		for (c = 1; c <= N; c++) {
			if (A[r][c] == -1 || A[r][c] == -2 || A[r][c] == 0) continue;
			reset_visited();
			vector<int> ret = get_size(r, c, A[r][c]);
			if (ret[0] + ret[1] < 2) continue;
			if (mx[0] + mx[1] < ret[0] + ret[1] || (mx[0] + mx[1] == ret[0] + ret[1] && mx[0] <= ret[0])) {
				mx_r = r; mx_c = c; mx[0] = ret[0]; mx[1] = ret[1];
			}
		}
	}

	if (mx_r == -1) return 0;

	fill_zero(mx_r, mx_c, A[mx_r][mx_c]);

	return (mx[0] + mx[1]) * (mx[0] + mx[1]);
}

void apply_gravity() {
	int r, c, x;

	for (c = 1; c <= N; c++) {
		for (r = N - 1; r >= 1; r--) {
			if (A[r][c] == -1 || A[r][c] == -2) continue;

			for (x = r + 1; x <= N; x++) {
				if (A[x][c] != -2) break;
			}
			x--;
			if (r != x) {
				A[x][c] = A[r][c];
				A[r][c] = -2;
			}
		}
	}
}

void rotate_blocks() {
	int r, c;
	for (r = 1; r <= N; r++)
		for (c = 1; c <= N; c++)
			B[N - c + 1][r] = A[r][c];

	for (r = 1; r <= N; r++)
		for (c = 1; c <= N; c++)
			A[r][c] = B[r][c];
}

int main() {
	int r, c;

	scanf("%d%d", &N, &M);
	for (r = 1; r <= N; r++)
		for (c = 1; c <= N; c++)
			scanf("%d", &A[r][c]);

	int ans = 0;
	while (1) {
		int ret = find_erase_bg();
		if (ret == 0) break;
		ans += ret;

		apply_gravity();

		rotate_blocks();

		apply_gravity();
	}
	printf("%d", ans);
}