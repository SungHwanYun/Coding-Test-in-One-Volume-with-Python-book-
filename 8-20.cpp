#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef pair<int, int> pii;

int n;
vector<pii> edges;
int A[100004][2];
vector<int> E[100004];
ll D[100004][2];

ll solve(int u, int color) {
    if (D[u][color] != -1) return D[u][color];
    ll& ret = D[u][color];
    ret = A[u][color];

    for (auto v : E[u]) {
        if (color == 0) {
            ret += min(solve(v, 0), solve(v, 1));
        }
        else {
            ret += solve(v, 0);
        }
    }
    return ret;
}

ll solution() {
    for (auto& e : edges) {
        int p = e.first, c = e.second;
        E[p].push_back(c);
    }
    for (int i = 0; i < n; i++) D[i][0] = D[i][1] = -1;
    return min(solve(0, 0), solve(0, 1));
}

int main() {
    ios_base::sync_with_stdio(0); cin.tie(0);
    cin >> n;
    for (int i = 1; i < n; i++) {
        int a, b; cin >> a >> b;
        edges.emplace_back(a, b);
    }
    for (int i = 0; i < n; i++) {
        cin >> A[i][0] >> A[i][1];
    }
    cout << solution();
}