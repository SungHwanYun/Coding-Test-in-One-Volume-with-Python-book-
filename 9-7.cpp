#include <bits/stdc++.h>
using namespace std;

int n = 0;
vector<vector<int>> E(102);
vector<int> L;
vector<int> X(102, -1);
vector<int> Y;

vector<int> assign_stone(int k, vector<int>& Y, vector<int>& target) {
    vector<int> cnt(n, 0);
    for (int i = 0; i < k; i++) {
        cnt[Y[i]] += 1;
    }
    vector<int> answer;
    for (int i = 0; i < k; i++) {
        int u = Y[i];
        cnt[u] -= 1;
        if (target[u] - 1 <= cnt[u] * 3) {
            answer.push_back(1);
            target[u] -= 1;
        }
        else if (target[u] - 2 <= cnt[u] * 3) {
            answer.push_back(2);
            target[u] -= 2;
        }
        else {
            answer.push_back(3);
            target[u] -= 3;
        }
    }
    return answer;
}

bool is_ok(int k, vector<int>& Y, vector<int>& target) {
    vector<int> cnt(n, 0);
    for (int i = 0; i < k; i++) {
        cnt[Y[i]] += 1;
    }
    for (int i = 0; i < n; i++) {
        if (target[i] < cnt[i] || cnt[i] * 3 < target[i]) {
            return false;
        }
    }
    return true;
}

int dfs(int u) {
    if (E[u].empty()) {
        return u;
    }
    int ret = dfs(E[u][X[u]]);
    X[u] = (X[u] + 1) % E[u].size();
    return ret;
}

vector<int> solution(vector<vector<int>> edges, vector<int> target) {
    n = edges.size() + 1;
    for (auto& edge : edges) {
        E[edge[0] - 1].push_back(edge[1] - 1);
    }
    for (int i = 0; i < n; i++) {
        sort(E[i].begin(), E[i].end());
        if (E[i].empty()) {
            L.push_back(i);
        }
        else {
            X[i] = 0;
        }
    }
    for (int i = 0; i < 10004; i++) {
        Y.push_back(dfs(0));
    }
    vector<int> answer = { -1 };
    for (int k = 1; k < 10004; k++) {
        if (is_ok(k, Y, target)) {
            answer = assign_stone(k, Y, target);
            break;
        }
    }
    return answer;
}