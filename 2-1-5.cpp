#include<bits/stdc++.h>
using namespace std;
int main() {
    int n; cin >> n;
    vector<vector<int>> A(n, vector<int>(n));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cin >> A[i][j];
        }
    }

    int i1, j1, i2, j2, k; cin >> i1 >> j1 >> i2 >> j2 >> k;
    for (int i = i1; i <= i2; i++) {
        for (int j = j1; j <= j2; j++) {
            A[i][j] *= k;
        }
    }

    int answer = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            answer += A[i][j];
        }
    }
    cout << answer;
}