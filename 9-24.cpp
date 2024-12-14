#include <bits/stdc++.h>
using namespace std;

enum DIR {
	LEFT = 1, LEFT_UP = 2, UP = 3, RIGHT_UP = 4, RIGHT = 5, RIGHT_DOWN = 6, DOWN = 7, LEFT_DOWN = 8
};

int dr[9] = { 0,0,-1,-1,-1,0,1,1,1 }, dc[9] = { 0,-1,-1,0,1,1,1,0,-1 };
int N = 4, Sr, Sc;
int A[5][5][10], C[5][5][104], X[5][5][10];

bool in_range(int r, int c) {
	return 1 <= r && r <= N && 1 <= c && c <= N;
}

void copy_A() {
	for (int r = 1; r <= N; r++) 
		for (int c = 1; c <= N; c++) 
			for (int d = 0; d <= 8; d++)
				X[r][c][d] = A[r][c][d];
}

void paste_A() {
	for (int r = 1; r <= N; r++) 
		for (int c = 1; c <= N; c++) 
			for (int d = 0; d <= 8; d++)
				A[r][c][d] += X[r][c][d];
}

int is_smell(int step, int r, int c) {
	if (step == 1) return 0;

	if (step == 2) return C[r][c][1];

	return C[r][c][step - 1] | C[r][c][step - 2]; 
}

void move_fish(int step) {
	int r, c, d, i;

	int T[5][5][10];
	for (r = 1; r <= N; r++)
		for (c = 1; c <= N; c++)
			for (d = 0; d <= 8; d++)
				T[r][c][d] = 0;

	for (r = 1; r <= N; r++)
		for (c = 1; c <= N; c++)
			for (d = 1; d <= 8; d++) {
				if (A[r][c][d] == 0) continue;
				int nr, nc, nd;.
				for (i = 0; i < 8; i++) {
					nd = d - i;
					if (nd <= 0) nd += 8;
					nr = r + dr[nd], nc = c + dc[nd];
					if (in_range(nr, nc) && !(nr == Sr && nc == Sc) && !is_smell(step, nr, nc))
						break;
				}
				if (i == 8) {
					nr = r; nc = c; nd = d;
				}
				T[nr][nc][nd] += A[r][c][d];
				T[nr][nc][0] += A[r][c][d];
			}

	for (r = 1; r <= N; r++)
		for (c = 1; c <= N; c++)
			for (d = 0; d <= 8; d++)
				A[r][c][d] = T[r][c][d];
}

void reset_A(int r, int c) {
	for (int d = 0; d <= 8; d++)
		A[r][c][d] = 0;
}

int rr[4] = { -1,0,1,0 }, cc[4] = { 0,-1,0,1 };

int get_sum(int d1, int d2, int d3) {
	int r, c, visited[5][5] = { 0, };
	visited[Sr + rr[d1]][Sc + cc[d1]] = 1;
	visited[Sr + rr[d1] + rr[d2]][Sc + cc[d1] + cc[d2]] = 1;
	visited[Sr + rr[d1] + rr[d2] + rr[d3]][Sc + cc[d1] + cc[d2] + cc[d3]] = 1;

	int ret = 0;
	for (r = 1; r <= N; r++)
		for (c = 1; c <= N; c++)
			if (visited[r][c])
				ret += A[r][c][0];
	return ret;
}

void move_shark(int step) {
	int i, j, k, x, y, z, mx = -1, r, c;

	for (i = 0; i < 4; i++) {
		r = Sr + rr[i]; c = Sc + cc[i];
		if (!in_range(r, c)) continue;
		for (j = 0; j < 4; j++) {
			r = Sr + rr[i] + rr[j]; c = Sc + cc[i] + cc[j];
			if (!in_range(r, c)) continue;
			for (k = 0; k < 4; k++) {
				r = Sr + rr[i] + rr[j] + rr[k]; c = Sc + cc[i] + cc[j] + cc[k];
				if (!in_range(r, c)) continue;
				int sum = get_sum(i, j, k);
				if (sum > mx) {
					mx = sum;
					x = i; y = j; z = k;
				}
			}
		}
	}

	r = Sr + rr[x]; c = Sc + cc[x];
	if (A[r][c][0] > 0) {
		reset_A(r, c);
		C[r][c][step] = 1;
	}

	r += rr[y]; c += cc[y];
	if (A[r][c][0] > 0) {
		reset_A(r, c);
		C[r][c][step] = 1;
	}

	r += rr[z]; c += cc[z];
	if (A[r][c][0] > 0) {
		reset_A(r, c);
		C[r][c][step] = 1;
	}

	Sr = r; Sc = c;
}

int main() {
	int m, r, c, d, s;

	scanf("%d %d", &m, &s);
	while (m-- > 0) {
		scanf("%d%d%d", &r, &c, &d);
		A[r][c][d]++;
	}
	scanf("%d%d", &Sr, &Sc);

	for (r = 1; r <= N; r++)
		for (c = 1; c <= N; c++)
			for (d = 1; d <= 8; d++)
				A[r][c][0] += A[r][c][d];

	for (int step = 1; step <= s; step++) {
		copy_A();

		move_fish(step);

		move_shark(step);

		paste_A();
	}

	int ans = 0;
	for (r = 1; r <= N; r++)
		for (c = 1; c <= N; c++)
			ans += A[r][c][0];
	printf("%d", ans);
}