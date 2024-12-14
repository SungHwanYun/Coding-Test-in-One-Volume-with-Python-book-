#include <iostream>
#include <vector>
using namespace std;

void do_add_query(vector<vector<long long>>& A, int i1, int j1, int i2, int j2, int k) {
    A[i1][j1] += k;
    if (j2 + 1 < A.size()) {
        A[i1][j2 + 1] -= k;
    }
    if (i2 + 1 < A.size()) {
        A[i2 + 1][j1] -= k;
    }
    if (i2 + 1 < A.size() && j2 + 1 < A.size()) {
        A[i2 + 1][j2 + 1] += k;
    }
}

long long get_sum(vector<vector<long long>>& A, int i1, int j1, int i2, int j2) {
    long long ret = 0;
    for (int i = i1; i <= i2; i++) {
        for (int j = j1; j <= j2; j++) {
            ret += A[i][j];
        }
    }
    return ret;
}

void solution(int n, vector<vector<long long>>& A, int m, vector<vector<int>>& Q) {
    vector<vector<long long>> psum(n, vector<long long>(n, 0));
    for (auto q : Q) {
        if (q[0] == 1) {
            do_add_query(psum, q[1], q[2], q[3], q[4], q[5]);
        }
        else {
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
            cout << get_sum(A, q[1], q[2], q[3], q[4]) + get_sum(psum, q[1], q[2], q[3], q[4]) << endl;
        }
    }
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int n, m;
    cin >> n >> m;
    vector<vector<long long>> A(n, vector<long long>(n, 0));
    vector<vector<int>> Q(m, vector<int>(6, 0));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cin >> A[i][j];
        }
    }
    for (int i = 0; i < m; i++) {
        cin >> Q[i][0];
        int t;
        if (Q[i][0] == 1) t = 6;
        else t = 5;
        for (int j = 1; j < t; j++) {
            cin >> Q[i][j];
        }
    }
    solution(n, A, m, Q);
    return 0;
}