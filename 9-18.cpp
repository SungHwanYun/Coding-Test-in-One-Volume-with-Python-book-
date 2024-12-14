#include <bits/stdc++.h>
using namespace std;

int E[204][204], D[204][204];
const int INF = (int)1e8;

int solution(int n, int s, int a, int b, vector<vector<int>> fares) {
    int i, j, k;
    for (i = 1; i <= n; i++)
        for (j = 1; j <= n; j++)
            if (i != j) E[i][j] = INF;

    for (int i = 0; i < fares.size(); i++) {
        int u = fares[i][0], v = fares[i][1], w = fares[i][2];
        E[u][v] = E[v][u] = w;
    }

    for (i = 1; i <= n; i++)
        for (j = 1; j <= n; j++)
            D[i][j] = E[i][j];
    for (k = 1; k <= n; k++)
        for (i = 1; i <= n; i++)
            for (j = 1; j <= n; j++)
                if (D[i][k] + D[k][j] < D[i][j])
                    D[i][j] = D[i][k] + D[k][j];

    int answer = D[s][a] + D[s][b];
    for (k = 1; k <= n; k++) {
        if (s == k) continue;
        int ret = D[s][k] + D[k][a] + D[k][b];
        answer = min(answer, ret);
    }
    return answer;
}