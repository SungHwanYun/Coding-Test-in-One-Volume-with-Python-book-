#include <iostream>
#include <vector>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int n, m;
    cin >> n >> m;

    vector<vector<long long>> A(n, vector<long long>(n));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cin >> A[i][j];
        }
    }

    vector<vector<int>> Q(m, vector<int>(6));
    for (int i = 0; i < m; i++) {
        cin >> Q[i][0];
        int t = 0;
        if (Q[i][0] == 1) t = 6;
        else t = 5;
        for (int j = 1; j < t; j++) {
            cin >> Q[i][j];
        }
    }

    vector<vector<long long>> psum(n, vector<long long>(n));
    bool psum_flag = false;

    for (int q = 0; q < m; q++) {
        if (Q[q][0] == 1) {
            int i1 = Q[q][1];
            int j1 = Q[q][2];
            int i2 = Q[q][3];
            int j2 = Q[q][4];
            int k = Q[q][5];

            psum[i1][j1] += k;
            if (j2 + 1 < n) {
                psum[i1][j2 + 1] -= k;
            }
            if (i2 + 1 < n) {
                psum[i2 + 1][j1] -= k;
            }
            if (i2 + 1 < n && j2 + 1 < n) {
                psum[i2 + 1][j2 + 1] += k;
            }
        }
        else {
            if (psum_flag == false) {
                psum_flag = true;
                for (int r = 0; r < n; r++) {
                    for (int c = 1; c < n; c++) {
                        psum[r][c] += psum[r][c - 1];
                    }
                }
                for (int c = 0; c < n; c++) {
                    for (int r = 1; r < n; r++) {
                        psum[r][c] += psum[r - 1][c];
                    }
                }
                for (int r = 0; r < n; r++) {
                    for (int c = 0; c < n; c++) {
                        A[r][c] += psum[r][c];
                    }
                }
                psum[0][0] = A[0][0];
                for (int c = 1; c < n; c++) {
                    psum[0][c] = psum[0][c - 1] + A[0][c];
                }
                for (int r = 1; r < n; r++) {
                    psum[r][0] = psum[r - 1][0] + A[r][0];
                }
                for (int r = 1; r < n; r++) {
                    for (int c = 1; c < n; c++) {
                        psum[r][c] = psum[r - 1][c] + psum[r][c - 1] - psum[r - 1][c - 1] + A[r][c];
                    }
                }
            }
            int i1 = Q[q][1];
            int j1 = Q[q][2];
            int i2 = Q[q][3];
            int j2 = Q[q][4];

            long long ret = psum[i2][j2];
            if (i1 > 0) {
                ret -= psum[i1 - 1][j2];
            }
            if (j1 > 0) {
                ret -= psum[i2][j1 - 1];
            }
            if (i1 > 0 && j1 > 0) {
                ret += psum[i1 - 1][j1 - 1];
            }

            cout << ret << "\n";
        }
    }
}