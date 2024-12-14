#include <bits/stdc++.h>
using namespace std;

enum DIR {
	RIGHT = 1, LEFT = 2, UP = 3, DOWN = 4
};
enum DIR_MASK {
	R_MASK = (1 << RIGHT), L_MASK = (1 << LEFT), U_MASK = (1 << UP), D_MASK = (1 << DOWN)
};
int dr[5] = { 0,0,0,-1,1 }, dc[5] = { 0,1,-1,0,0 };
int N, M, K, A[24][24], B[24][24], C[24][24], W[24][24];
struct qinfo {
	int r, c, t;
	qinfo(int r1, int c1, int t1) : r(r1), c(c1), t(t1) {}
};

bool in_range(int r, int c) {
	return 1 <= r && r <= N && 1 <= c && c <= M;
}

void build_B_sub(int sr, int sc, int d) {
	int r, c, visited[24][24];

	for (r = 1; r <= N; r++) 
		for (c = 1; c <= M; c++) 
			visited[r][c] = 0;

	queue<qinfo> Q;

	Q.push(qinfo(sr + dr[d], sc + dc[d], 5));
	visited[sr + dr[d]][sc + dc[d]] = 1;

	while (!Q.empty()) {
		int nr, nc;
		qinfo now = Q.front(); Q.pop();
		r = now.r; c = now.c;

		B[r][c] += now.t;

		if (now.t == 1) continue;

		if (d == RIGHT) {
			nr = r - 1; nc = c + 1;
			if (in_range(nr, nc) && visited[nr][nc] == 0 && !(W[r][c] & U_MASK) && !(W[nr][nc] & L_MASK)) {
				Q.push(qinfo(nr, nc, now.t - 1));
				visited[nr][nc] = 1;
			}

			nr = r;
			if (in_range(nr, nc) && visited[nr][nc] == 0 && !(W[r][c] & R_MASK)) {
				Q.push(qinfo(nr, nc, now.t - 1));
				visited[nr][nc] = 1;
			}

			nr = r + 1;
			if (in_range(nr, nc) && visited[nr][nc] == 0 && !(W[r][c] & D_MASK) && !(W[nr][nc] & L_MASK)) {
				Q.push(qinfo(nr, nc, now.t - 1));
				visited[nr][nc] = 1;
			}
		}
		else if (d == LEFT) {
			nr = r - 1; nc = c - 1;
			if (in_range(nr, nc) && visited[nr][nc] == 0 && !(W[r][c] & U_MASK) && !(W[nr][nc] & R_MASK)) {
				Q.push(qinfo(nr, nc, now.t - 1));
				visited[nr][nc] = 1;
			}

			nr = r;
			if (in_range(nr, nc) && visited[nr][nc] == 0 && !(W[r][c] & L_MASK)) {
				Q.push(qinfo(nr, nc, now.t - 1));
				visited[nr][nc] = 1;
			}

			nr = r + 1;
			if (in_range(nr, nc) && visited[nr][nc] == 0 && !(W[r][c] & D_MASK) && !(W[nr][nc] & R_MASK)) {
				Q.push(qinfo(nr, nc, now.t - 1));
				visited[nr][nc] = 1;
			}
		}
		else if (d == UP) {
			nr = r - 1; nc = c - 1;
			if (in_range(nr, nc) && visited[nr][nc] == 0 && !(W[r][c] & L_MASK) && !(W[nr][nc] & D_MASK)) {
				Q.push(qinfo(nr, nc, now.t - 1));
				visited[nr][nc] = 1;
			}

			nc = c;
			if (in_range(nr, nc) && visited[nr][nc] == 0 && !(W[r][c] & U_MASK)) {
				Q.push(qinfo(nr, nc, now.t - 1));
				visited[nr][nc] = 1;
			}

			nc = c + 1;
			if (in_range(nr, nc) && visited[nr][nc] == 0 && !(W[r][c] & R_MASK) && !(W[nr][nc] & D_MASK)) {
				Q.push(qinfo(nr, nc, now.t - 1));
				visited[nr][nc] = 1;
			}
		}
		else {
			nr = r + 1; nc = c - 1;
			if (in_range(nr, nc) && visited[nr][nc] == 0 && !(W[r][c] & L_MASK) && !(W[nr][nc] & U_MASK)) {
				Q.push(qinfo(nr, nc, now.t - 1));
				visited[nr][nc] = 1;
			}

			nc = c;
			if (in_range(nr, nc) && visited[nr][nc] == 0 && !(W[r][c] & D_MASK)) {
				Q.push(qinfo(nr, nc, now.t - 1));
				visited[nr][nc] = 1;
			}

			nc = c + 1;
			if (in_range(nr, nc) && visited[nr][nc] == 0 && !(W[r][c] & R_MASK) && !(W[nr][nc] & U_MASK)) {
				Q.push(qinfo(nr, nc, now.t - 1));
				visited[nr][nc] = 1;
			}
		}
	}
}

void build_B() {
	for (int r = 1; r <= N; r++) {
		for (int c = 1; c <= M; c++) {
			if (0 < C[r][c] && C[r][c] < 5) {
				build_B_sub(r, c, C[r][c]);
			}
		}
	}
}

int main() {
	int i, j, w, r, c, t;
	scanf("%d%d%d", &N, &M, &K);
	for (i = 1; i <= N; i++)
		for (j = 1; j <= M; j++)
			scanf("%d", &C[i][j]);

	scanf("%d", &w);
	while (w-- > 0) {
		scanf("%d%d%d", &r, &c, &t);
		if (t == 0) {
			W[r][c] |= U_MASK;
			W[r - 1][c] |= D_MASK;
		}
		else {
			W[r][c] |= R_MASK;
			W[r][c + 1] |= L_MASK;
		}
	}

	build_B();

	for (int step = 1; step <= 100; step++) {
		for (r = 1; r <= N; r++)
			for (c = 1; c <= M; c++)
				A[r][c] += B[r][c];

		int X[24][24];
		for (r = 1; r <= N; r++)
			for (c = 1; c <= M; c++)
				X[r][c] = 0;
		for (r = 1; r <= N; r++) for (c = 1; c <= M; c++) {
			int nr, nc, diff;
			nr = r; nc = c + 1;
			if (in_range(nr, nc) && !(W[r][c] & R_MASK)) {
				diff = (A[r][c] - A[nr][nc]) / 4;
				X[r][c] -= diff;
				X[nr][nc] += diff;
			}

			nr = r + 1; nc = c;
			if (in_range(nr, nc) && !(W[r][c] & D_MASK)) {
				diff = (A[r][c] - A[nr][nc]) / 4;
				X[r][c] -= diff;
				X[nr][nc] += diff;
			}
		}
		for (r = 1; r <= N; r++) for (c = 1; c <= M; c++)
			A[r][c] += X[r][c];

		for (c = 1; c <= M; c++) {
			if (A[1][c] >= 1) A[1][c]--;
			if (A[N][c] >= 1) A[N][c]--;
		}
		for (r = 2; r < N; r++) {
			if (A[r][1] >= 1) A[r][1]--;
			if (A[r][M] >= 1) A[r][M]--;
		}

		int is_ok = 1;
		for (r = 1; r <= N; r++) for (c = 1; c <= M; c++) {
			if (C[r][c] == 5 && A[r][c] < K) is_ok = 0;
		}
		if (is_ok) {
			printf("%d", step);
			return 0;
		}
	}
	printf("101");
}