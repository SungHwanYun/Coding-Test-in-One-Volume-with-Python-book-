#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef pair<int, int> pii;

int solve(int u, int k, vector<int>& A, vector<vector<int>>& E) {
	int ret = A[u];
	if (k == 0) return ret;
	for (auto& v : E[u]) {
		ret += solve(v, k - 1, A, E);
	}
	return ret;
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
	cout << solve(0, k, A, E);
}