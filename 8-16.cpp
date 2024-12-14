#include <bits/stdc++.h>
using namespace std;

int solution(vector<vector<int>>& A) {
    vector<vector<int>> D(6, vector<int>(2, 0));
    D[0][0] = D[0][1] = A[0][1];
    for (int i = 1; i < 6; i++) {
        D[i][0] = min(D[i - 1][0] + A[2 * i - 2][2 * i + 1] + A[2 * i][2 * i + 1], \
            D[i - 1][1] + A[2 * i - 1][2 * i + 1] + A[2 * i][2 * i + 1]);
        D[i][1] = min(D[i - 1][0] + A[2 * i - 2][2 * i] + A[2 * i][2 * i + 1], \
            D[i - 1][1] + A[2 * i - 1][2 * i] + A[2 * i][2 * i + 1]);
    }
    return min(D[5][0], D[5][1]);
}

int main() {
    vector<vector<int>> A(12, vector<int>(12, 0));
    for (int i = 0; i < 12; i++) {
        for (int j = 0; j < 12; j++) {
            cin >> A[i][j];
        }
    }
    cout << solution(A) << endl;
}