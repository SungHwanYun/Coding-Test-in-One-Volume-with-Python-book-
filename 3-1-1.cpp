#include<bits/stdc++.h>
using namespace std;
int main() {
    int n, m; cin >> n >> m;
    vector<long long> A(n);
    for (auto& a : A)
        cin >> a;
    for (int i = 0; i < m; i++) {
        long long k; cin >> k;
        int cnt = 0;
        for (int j = 0; j < n; j++) {
            if (A[j] >= k) cnt++;
        }
        cout << cnt << '\n';
    }
}