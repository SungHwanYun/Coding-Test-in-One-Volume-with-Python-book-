#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
void do_add_query(int psum[], int i, int j, int k, int n) {
    psum[i] += k;
    if (j + 1 < n) psum[j + 1] -= k;
}
ll get_sum(ll psum2[], int i, int j) {
    ll ret = psum2[j];
    if (i > 0) ret -= psum2[i - 1];
    return ret;
}
int A[100004];
int psum[100004];
ll psum2[100004];
int main() {
    ios::sync_with_stdio(false);
    cin.tie(NULL);
    cout.tie(NULL);

    int n, m; cin >> n >> m;
    for (int i = 0; i < n; i++) cin >> A[i];
    int psum_flag = 0;
    while (m--) {
        int op, i, j, k; cin >> op;
        if (op == 1) {
            cin >> i >> j >> k;
            do_add_query(psum, i, j, k, n);
        }
        else {
            cin >> i >> j;
            if (psum_flag == 0) {
                psum_flag = 1;
                for (int t = 1; t < n; t++) psum[t] += psum[t - 1];
                for (int t = 0; t < n; t++) A[t] += psum[t];
                psum2[0] = A[0];
                for (int t = 1; t < n; t++) psum2[t] = psum2[t - 1] + A[t];
            }
            cout << get_sum(psum2, i, j) << '\n';
        }
    }
}