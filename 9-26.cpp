#include <bits/stdc++.h>
using namespace std;

int dr[4] = { 0,0,-1,1 }, dc[4] = { -1,1,0,0 };
int point[5] = { 0,1,10,100,1000 };

int N, A[21][21], B[401][4];

bool in_range(int r, int c) {
	return 1 <= r && r <= N && 1 <= c && c <= N;
}

void set_student(int x, int y1, int y2, int y3, int y4) {
	int r, c, i;
	int mx_r = 1, mx_c = 1, mx_empty = -1, mx_favorate = -1;

	for (r = 1; r <= N; r++) {
		for (c = 1; c <= N; c++) {
			if (A[r][c] != 0) continue;
			int empty = 0, favorate = 0;
			for (i = 0; i < 4; i++) {
				int nr = r + dr[i], nc = c + dc[i];
				if (in_range(nr, nc) == false) continue;
				if (A[nr][nc] == 0) empty++;
				else if (A[nr][nc] == y1 || A[nr][nc] == y2 || A[nr][nc] == y3 || A[nr][nc] == y4) favorate++;
			}
			if (mx_favorate < favorate || (mx_favorate == favorate && mx_empty < empty)) {
				mx_r = r; mx_c = c; mx_empty = empty; mx_favorate = favorate;
			}
		}
	}
	A[mx_r][mx_c] = x;
}
int main() {
	int i, r, c;
	scanf("%d", &N);
	for (i = 0; i < N * N; i++) {
		int x, y1, y2, y3, y4;
		scanf("%d%d%d%d%d", &x, &y1, &y2, &y3, &y4);
		B[x][0] = y1; B[x][1] = y2; B[x][2] = y3; B[x][3] = y4;

		set_student(x, y1, y2, y3, y4);
	}

	int ans = 0;
	for (r = 1; r <= N; r++)
		for (c = 1; c <= N; c++) {
			int student = A[r][c], favorate = 0;
			for (i = 0; i < 4; i++) {
				int nr = r + dr[i], nc = c + dc[i];
				if (in_range(nr, nc) == false) continue;
				else if (A[nr][nc] == B[student][0] || A[nr][nc] == B[student][1] || A[nr][nc] == B[student][2] || A[nr][nc] == B[student][3]) favorate++;
			}
			ans += point[favorate];
		}
	printf("%d", ans);
}