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
        ll k; cin >> k;
        int i = lower_bound(A.begin(), A.end(), k) - A.begin();
        cout << n - i << '\n';
    }
}