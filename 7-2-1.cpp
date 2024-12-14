#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef pair<int, int> pii;

int solve(int u, int depth, int k, vector<int>& A, vector<vector<int>>& E) {
	if (A[u] == k) return depth;
	for (auto& v : E[u]) {
		int ret = solve(v, depth + 1, k, A, E);
		if (ret != -1) return ret;
	}
	return -1;
}
int main(int argc, char* argv[]) {
	ios::sync_with_stdio(false);
	cin.tie(NULL);
	cout.tie(NULL);

	int n, k; cin >> n >> k;
	vector<vector<int>>E(n);
	for (int i = 1; i < n; i++) {
		int u, v; cin >> u >> v;
		E[u].push_back(v);
	}
	vector<int>A(n);
	for (auto& a : A) cin >> a;

	cout << solve(0, 0, k, A, E);
}