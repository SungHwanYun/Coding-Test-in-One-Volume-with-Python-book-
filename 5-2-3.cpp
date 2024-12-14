#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
int main() {
    ios::sync_with_stdio(false);
    cin.tie(NULL);
    cout.tie(NULL);

    int n, m; cin >> n >> m;
    vector<ll> A(n);
    for (auto& a : A) cin >> a;
    sort(A.begin(), A.end());
    while (m--) {
        ll i, j; cin >> i >> j;
        int ret = upper_bound(A.begin(), A.end(), j) - lower_bound(A.begin(), A.end(), i);
        cout << ret << '\n';
    }
}