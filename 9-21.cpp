#include <bits/stdc++.h>
using namespace std;

vector<int> E[300004];
int D[300004][2];

void solve(int r, vector<int>& sales) {
    int child_sum = 0;
    int diff_mn = (int)2e9;
    int is_zero_larger = 0;

    for (int i = 0; i < E[r].size(); i++) {
        int c = E[r][i];
        solve(c, sales);
        child_sum += min(D[c][0], D[c][1]);
        if (D[c][0] >= D[c][1]) {
            is_zero_larger = 1;
        }

        if (D[c][0] <= D[c][1]) {
            diff_mn = min(diff_mn, D[c][1] - D[c][0]);
        }
    }

    D[r][1] = child_sum + sales[r - 1];

    if (E[r].size() == 0) {
        D[r][0] = 0;
    }
    else if (is_zero_larger == 1) {
        D[r][0] = child_sum;
    }
    else {
        D[r][0] = child_sum + diff_mn;
    }
}

int solution(vector<int> sales, vector<vector<int>> links) {
    for (int i = 0; i < links.size(); i++) {
        int p = links[i][0], c = links[i][1];
        E[p].push_back(c);
    }
    solve(1, sales);

    return min(D[1][0], D[1][1]);
}