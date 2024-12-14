#include <bits/stdc++.h>
using namespace std;

typedef pair<int, int> pii;

int dr[9] = { 0,0,-1,-1,-1,0,1,1,1 }, dc[9] = { 0,-1,-1,0,1,1,1,0,-1 };
int N, M, A[50][50], C[50][50];
vector<pii> B;

bool in_range(int r, int c) {
	return 0 <= r && r < N && 0 <= c && c < N;
}

void move_cloud(int dd, int ss) {
	int i, r, c, nr, nc;
	vector<pii> V;
	for (i = 0; i < B.size(); i++) {
		r = B[i].first; c = B[i].second;
		nr = (r + dr[dd] * ss) % N; nc = (c + dc[dd] * ss) % N;
		if (nr < 0) nr += N;
		if (nc < 0) nc += N;
		V.push_back({ nr,nc });
	}
	B.clear();
	B.assign(V.begin(), V.end());
}

void add_one_water() {
	int i, r, c;

	for (r = 0; r < N; r++) for (c = 0; c < N; c++) C[r][c] = 0;
	for (i = 0; i < B.size(); i++) {
		r = B[i].first; c = B[i].second;
		C[r][c] = 1;
		A[r][c]++;
	}

	for (i = 0; i < B.size(); i++) {
		r = B[i].first; c = B[i].second;
		int rr[4] = { -1,-1,1,1 }, cc[4] = { -1,1,-1,1 };
		for (int j = 0; j < 4; j++) {
			int nr = r + rr[j], nc = c + cc[j];
			if (in_range(nr, nc) && A[nr][nc] > 0)
				A[r][c]++;
		}
	}

	B.clear();
}

void build_cloud() {
	int r, c;
	for (r = 0; r < N; r++)
		for (c = 0; c < N; c++) {
			if (A[r][c] >= 2 && C[r][c] == 0) {
				B.push_back({ r,c });
				A[r][c] -= 2;
			}
		}
}

int main() {
	int r, c, d, s;

	scanf("%d%d", &N, &M);
	for (r = 0; r < N; r++)
		for (c = 0; c < N; c++)
			scanf("%d", &A[r][c]);

	B.push_back({ N - 1,0 }); B.push_back({ N - 1,1 });
	B.push_back({ N - 2,0 }); B.push_back({ N - 2,1 });

	while (M-- > 0) {
		scanf("%d%d", &d, &s);

		move_cloud(d, s);

		add_one_water();

		build_cloud();
	}
	int ans = 0;
	for (r = 0; r < N; r++)
		for (c = 0; c < N; c++)
			ans += A[r][c];
	printf("%d", ans);
}