#include<bits/stdc++.h>
using namespace std;

vector<int> E[20];
int N, answer;

void solve(int state, int sheep, int wolf, vector<int>& info) {
    if (sheep > answer)
        answer = sheep;

    for (int u = 0; u < N; u++) {
        if ((state & (1 << u)) == 0) continue;

        for (int i = 0; i < E[u].size(); i++) {
            int v = E[u][i];
            if (state & (1 << v)) continue;

            if (info[v] == 0) {
                solve(state | (1 << v), sheep + 1, wolf, info);
            }
            else {
                if (sheep > wolf + 1) {
                    solve(state | (1 << v), sheep, wolf + 1, info);
                }
            }
        }
    }
}

int solution(vector<int> info, vector<vector<int>> edges) {
    N = info.size();
    for (int i = 0; i < edges.size(); i++) {
        int p = edges[i][0], c = edges[i][1];
        E[p].push_back(c);
    }

    solve(1, 1, 0, info);
    return answer;
}