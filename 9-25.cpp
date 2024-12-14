#include <bits/stdc++.h>
using namespace std;

int N, K, A[100][100], B[100][100], S[100];

bool is_ok() {
	int i, j, mx = 0, mn = (int)1e9;
	for (i = 0; i < N; i++) {
		for (j = 0; j < S[i]; j++) {
			mx = max(mx, A[i][j]);
			mn = min(mn, A[i][j]);
		}
	}
	return mx - mn <= K;
}

void add_one_fish() {
	int i, mn = (int)1e9;
	for (i = 0; i < N; i++)
		mn = min(mn, A[i][0]);
	for (i = 0; i < N; i++)
		if (A[i][0] == mn)
			A[i][0]++;
}

void move_column(int src, int dst) {
	int i;
	for (i = 0; i < S[src]; i++)
		A[dst][i] = A[src][i];
	S[dst] = S[src];
}

void build_up_sub(int e) {
	int i, j, k;

	for (i = e; i >= 0; i--)
		for (j = e + 1, k = 0; k < S[i]; j++, k++)
			A[j][S[j]++] = A[i][k];

	for (i = 0, j = e + 1; j < N; i++, j++)
		move_column(j, i);

	N = N - e - 1;
}
void build_up() {
	while (1) {
		for (int i = 1; i < N; i++)
			if (S[i] == 1)
				break;

		if (S[0] > N - i) break;
		build_up_sub(i - 1);
	}
}

void adjust_fish_sub(int r1, int c1, int r2, int c2) {
	int diff;
	if (A[r1][c1] > A[r2][c2]) {
		diff = (A[r1][c1] - A[r2][c2]) / 5;
		B[r1][c1] -= diff;
		B[r2][c2] += diff;
	}
	else {
		diff = (A[r2][c2] - A[r1][c1]) / 5;
		B[r1][c1] += diff;
		B[r2][c2] -= diff;
	}
}
void adjust_fish() {
	int r, c;

	for (r = 0; r < N; r++)
		for (c = 0; c < S[r]; c++)
			B[r][c] = 0;

	for (r = 0; r < N; r++) {
		for (c = 0; c < S[r]; c++) {
			if (r < N - 1 && c < S[r + 1])
				adjust_fish_sub(r, c, r + 1, c);

			if (c + 1 < S[r])
				adjust_fish_sub(r, c, r, c + 1);
		}
	}

	for (r = 0; r < N; r++)
		for (c = 0; c < S[r]; c++)
			A[r][c] += B[r][c];
}

void spread_fish() {
	int i, j, k = 0;

	for (i = 0; i < N; i++)
		for (j = 0; j < S[i]; j++)
			B[k++][0] = A[i][j];

	N = k;
	for (i = 0; i < N; i++) {
		A[i][0] = B[i][0];
		S[i] = 1;
	}
}

void build_up_half() {
	int r1, c, r2, e = N / 2 - 1;

	for (c = S[0] - 1; c >= 0; c--)
		for (r1 = e, r2 = e + 1; r1 >= 0; r1--, r2++)
			A[r2][S[r2]++] = A[r1][c];

	for (r1 = 0, r2 = e + 1; r2 < N; r1++, r2++)
		move_column(r2, r1);
	N /= 2;
}

int main() {
	int i, step = 0;

	scanf("%d%d", &N, &K);
	for (i = 0; i < N; i++) {
		scanf("%d", &A[i][0]);
		S[i] = 1;
	}

	while (1) {
		if (is_ok()) break;

		add_one_fish();

		build_up();

		adjust_fish();

		spread_fish();

		build_up_half();
		build_up_half();

		adjust_fish();

		spread_fish();

		step++;
	}
	printf("%d", step);
}