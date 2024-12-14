#include <bits/stdc++.h>
using namespace std;

int solve(int state, int k, int apple, vector<int>& A, vector<vector<int>>& E, vector<int>& visited) {
    if (visited[state] == 1) {
        return 0;
    }
    visited[state] = 1;
    int ret = apple;
    if (k == 0) {
        return ret;
    }
    for (int u = 0; u < A.size(); u++) {
        if ((state & (1 << u)) == 0) {
            continue;
        }
        for (int v : E[u]) {
            if (state & (1 << v)) {
                continue;
            }
            ret = max(ret, solve(state | (1 << v), k - 1, apple + A[v], A, E, visited));
        }
    }
    return ret;
}

int solution(int n, int k, vector<int>& A, vector<vector<int>>& edges) {
    vector<vector<int>> E(n);
    for (auto& edge : edges) {
        int p = edge[0];
        int c = edge[1];
        E[p].push_back(c);
    }
    vector<int> visited(1 << n, 0);
    return solve(1 << 0, k - 1, A[0], A, E, visited);
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int n, k;
    cin >> n >> k;
    vector<vector<int>> edges(n - 1, vector<int>(2));
    for (int i = 0; i < n - 1; i++) {
        cin >> edges[i][0] >> edges[i][1];
    }
    vector<int> A(n);
    for (int i = 0; i < n; i++) {
        cin >> A[i];
    }
    cout << solution(n, k, A, edges) << endl;
}