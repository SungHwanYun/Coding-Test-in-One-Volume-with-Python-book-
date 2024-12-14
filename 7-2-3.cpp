#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef pair<int, int> pii;

ll solve(int u, vector<int>& A, vector<vector<int>>& E) {
	ll ret = A[u];
	for (auto& v : E[u]) {
		ll ret2 = solve(v, A, E);
		if (ret2 > 0) ret += ret2;
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
	vector<int>A(n);
	for (auto& a : A) cin >> a;
	cout << solve(0, A, E);
}