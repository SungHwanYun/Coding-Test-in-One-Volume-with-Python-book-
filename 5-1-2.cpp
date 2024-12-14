#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
void do_add_query(int psum[], int i, int j, int k, int n) {
    psum[i] += k;
    if (j + 1 < n) psum[j + 1] -= k;
}
ll get_sum(int A[], int psum[], int i, int j, int n) {
    for (int t = 1; t < n; t++) psum[t] += psum[t - 1];
    ll ret = 0;
    for (int t = i; t <= j; t++) ret += psum[t] + A[t];
    return ret;
}
int main() {
    ios::sync_with_stdio(false);
    cin.tie(NULL);
    cout.tie(NULL);

    int n, m; cin >> n >> m;
    int A[100004];
    for (int i = 0; i < n; i++) cin >> A[i];
    int psum[100004] = { 0 };
    while (m--) {
        int op, i, j, k; cin >> op;
        if (op == 1) {
            cin >> i >> j >> k;
            do_add_query(psum, i, j, k, n);
        }
        else {
            cin >> i >> j;
            cout << get_sum(A, psum, i, j, n) << '\n';
        }
    }
}