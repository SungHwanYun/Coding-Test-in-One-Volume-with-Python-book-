#include <bits/stdc++.h>
using namespace std;

vector<int> solve(int state, int k, int apple, int pear, vector<int>& A, vector<vector<int>>& E, vector<int>& visited) {
    if (visited[state] == 1) {
        return { 0, 0 };
    }
    visited[state] = 1;
    vector<int> ret = { apple, pear };
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
            vector<int> ret2 = solve(state | (1 << v), k - 1, apple + (A[v] == 1), pear + (A[v] == 2), A, E, visited);
            if (ret2[0] * ret2[1] > ret[0] * ret[1] ||
                (ret2[0] * ret2[1] == ret[0] * ret[1] && ret2[0] > ret[0]) ||
                (ret2[0] * ret2[1] == ret[0] * ret[1] && ret2[0] == ret[0] && ret2[1] > ret[1])) {
                ret[0] = ret2[0];
                ret[1] = ret2[1];
            }
        }
    }
    return ret;
}

vector<int> solution(int n, int k, vector<int>& A, vector<vector<int>>& edges) {
    vector<vector<int>> E(n);
    for (auto& edge : edges) {
        E[edge[0]].push_back(edge[1]);
    }
    vector<int> visited(1 << n, 0);
    return solve(1 << 0, k - 1, (A[0] == 1), (A[0] == 2), A, E, visited);
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
    vector<int> ret = solution(n, k, A, edges);
    cout << ret[0] << " " << ret[1] << endl;
}