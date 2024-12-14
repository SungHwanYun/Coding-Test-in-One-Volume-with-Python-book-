#include <bits/stdc++.h>
using namespace std;

int dr[4] = { 0,1,0,-1 }, dc[4] = { 1,0,-1,0 };
int A[24][24], B[24][24], C[6][6], N, M, K, visited[24][24];

bool in_range(int r, int c) {
	return 1 <= r && r <= N && 1 <= c && c <= M;
}

int dfs(int r, int c) {
	int ret = 1;
	visited[r][c] = 1;

	for (int i = 0; i < 4; i++) {
		int nr = r + dr[i], nc = c + dc[i];
		if (in_range(nr, nc) && visited[nr][nc] == 0 && A[r][c] == A[nr][nc])
			ret += dfs(nr, nc);
	}
	return ret;
}

void get_next_pos(int r, int c, int& nr, int& nc, int& d) {
	nr = r + dr[d]; nc = c + dc[d];
	if (in_range(nr, nc) == false) {
		d = (d + 2) % 4;
		nr = r + dr[d];
		nc = c + dc[d];
	}
}

void rotate_cube(int d) {
	int t;
	if (d == 0) {
		t = C[4][2];
		C[4][2] = C[2][3];
		C[2][3] = C[2][2];
		C[2][2] = C[2][1];
		C[2][1] = t;
	}
	else if (d == 2) {
		t = C[4][2];
		C[4][2] = C[2][1];
		C[2][1] = C[2][2];
		C[2][2] = C[2][3];
		C[2][3] = t;
	}
	else if (d == 1) {
		t = C[4][2];
		C[4][2] = C[3][2];
		C[3][2] = C[2][2];
		C[2][2] = C[1][2];
		C[1][2] = t;
	}
	else {
		t = C[4][2];
		C[4][2] = C[1][2];
		C[1][2] = C[2][2];
		C[2][2] = C[3][2];
		C[3][2] = t;
	}
}

void update_dir(int r, int c, int& d) {
	int a = C[4][2], b = A[r][c];

	if (a > b) {
		d = (d + 1) % 4;
	}
	else if (a < b) {
		d = (d - 1 + 4) % 4;
	}
}

int main() {
	int i, j, k;

	cin >> N >> M >> K;
	for (i = 1; i <= N; i++) {
		for (j = 1; j <= M; j++) {
			cin >> A[i][j];
		}
	}

	for (i = 1; i <= N; i++) {
		for (j = 1; j <= M; j++) {
			for (int r = 1; r <= N; r++)
				for (int c = 1; c <= M; c++)
					visited[r][c] = 0;
			B[i][j] = dfs(i, j) * A[i][j];
		}
	}

	C[1][2] = 2;
	C[2][1] = 4; C[2][2] = 1; C[2][3] = 3;
	C[3][2] = 5;
	C[4][2] = 6;

	int ans = 0, r = 1, c = 1, d = 0;
	for (k = 0; k < K; k++) {
		int nr, nc;
		get_next_pos(r, c, nr, nc, d);
		ans += B[nr][nc];

		rotate_cube(d);

		r = nr; c = nc;

		update_dir(r, c, d);
	}
	cout << ans;
}