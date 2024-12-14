#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef pair<int, int> pii;

ll solve(int u, int color, vector<vector<int>>& A, vector<vector<int>>& E) {
	ll ret = A[u][color];
	for (auto& v : E[u]) {
		ret += solve(v, 1 - color, A, E);
	}
	return ret;
}
int main(int argc, char* argv[]) {
	ios::sync_with_stdio(false);
	cin.tie(NULL);
	cout.tie(NULL);

	int n; cin >> n;
	vector<vector<int>>E(n);
	for (int i = 1; i < n; i++) {
		int u, v; cin >> u >> v;
		E[u].push_back(v);
	}
	vector<vector<int>>A(n);
	for (int i = 0; i < n; i++) {
		int w, b; cin >> w >> b;
		A[i].push_back(w); A[i].push_back(b);
	}
	cout << min(solve(0, 0, A, E), solve(0, 1, A, E));
}