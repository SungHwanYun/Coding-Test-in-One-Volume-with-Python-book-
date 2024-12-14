#include<bits/stdc++.h>
using namespace std;
int main() {
    ios::sync_with_stdio(false);
    cin.tie(NULL);
    cout.tie(NULL);
    int n, m; cin >> n >> m;
    vector<vector<int>>A(n, vector<int>(n));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cin >> A[i][j];
        }
    }

    while (m--) {
        int op; cin >> op;
        int i1, j1, i2, j2, k;
        cin >> i1 >> j1 >> i2 >> j2;
        if (op == 1) {
            cin >> k;
            for (int i = i1; i <= i2; i++) {
                for (int j = j1; j <= j2; j++) {
                    A[i][j] += k;
                }
            }
        }
        else {
            long long answer = 0;
            for (int i = i1; i <= i2; i++) {
                for (int j = j1; j <= j2; j++) {
                    answer += A[i][j];
                }
            }
            cout << answer << '\n';
        }
    }
}