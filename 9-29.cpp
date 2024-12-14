#include <bits/stdc++.h>
using namespace std;

enum DIR {
	LEFT = 0, DOWN = 1, RIGHT = 2, UP = 3
};

int dr[4] = { 0,1,0,-1 }, dc[4] = { -1,0,1,0 };
int N, M, A[50][50], B[50][50], C[2500], D[2500], ans[4];

bool in_range(int r, int c) {
	return 0 <= r && r < N && 0 <= c && c < N;
}

void build_BC() {
	int r, c, d, remain_move, total_move, step, cnt = 0;

	r = c = N / 2;
	d = LEFT;
	remain_move = total_move = 1;
	step = 1;

	while (1) {
		if (in_range(r, c) == false) break;

		B[r][c] = cnt;
		C[cnt] = A[r][c];
		cnt++;

		r = r + dr[d]; c = c + dc[d];
		remain_move--;

		if (remain_move == 0 && step == 2) {
			total_move++;
			remain_move = total_move;
			step = 1;
			d = (d + 1) % 4;
		}
		else if (remain_move == 0) {
			remain_move = total_move;
			step = 2;
			d = (d + 1) % 4;
		}
	}
}

void move_thing_slow() {
	int i, j, empty = 0;
	for (i = 1; i < N * N; i++) {
		if (C[i] == 0) {
			empty++;
			continue;
		}
		if (empty == 0) continue;

		for (j = 1; j < i; j++)
			if (C[j] == 0) break;
		C[j] = C[i];
		C[i] = 0;
	}
}

void move_thing_fast() {
	int i, j;
	for (i = 1, j = 1; i < N * N; i++) {
		if (C[i] != 0) {
			if (i == j) {
				j++;
			}
			else {
				C[j] = C[i];
				j++;
			}
		}
	}
}

int explode_thing_sub() {
	int i = 1, j, ret = 0;

	while (i < N * N) {
		if (C[i] == 0) {
			i++;
			continue;
		}

		for (j = i + 1; j < N * N; j++) {
			if (C[i] != C[j]) break;
		}
		if (j - i >= 4) {
			ans[C[i]] += j - i;
			for (int k = i; k < j; k++) C[k] = 0;
			ret = 1;
		}
		i = j;
	}
	return ret;
}
void explode_thing() {
	while (1) {
		int ret = explode_thing_sub();
		if (ret == 0) break;

		move_thing_slow();
	}
}

void change_thing() {
	int i = 1, j, k = 1;

	while (i < N * N) {
		if (C[i] == 0) {
			i++;
			continue;
		}
		if (k >= N * N - 1) break;

		for (j = i + 1; j < N * N; j++) {
			if (C[i] != C[j]) break;
		}
		D[k++] = j - i;
		D[k++] = C[i];
		i = j;
	}

	for (i = 0; i < k; i++)
		C[i] = D[i];
	for (i = k; i < N * N; i++)
		C[i] = 0;
}

int main() {
	int r, c, d, s;

	scanf("%d%d", &N, &M);
	for (r = 0; r < N; r++)
		for (c = 0; c < N; c++)
			scanf("%d", &A[r][c]);

	build_BC();

	while (M-- > 0) {
		scanf("%d%d", &d, &s);

		int rr[5] = { 0,-1,1,0,0 }, cc[5] = { 0,0,0,-1,1 };

		r = c = N / 2;
		while (s-- > 0) {
			r = r + rr[d]; c = c + cc[d];
			C[B[r][c]] = 0;
		}

		move_thing_slow();

		explode_thing();

		change_thing();
	}
	printf("%d", ans[1] + 2 * ans[2] + 3 * ans[3]);
}