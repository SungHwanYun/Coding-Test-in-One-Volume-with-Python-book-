#include<bits/stdc++.h>
using namespace std;
int main() {
    int n, m; cin >> n >> m;
    vector<string> A(n);
    for (auto& a : A)
        cin >> a;
    for (int i = 0; i < m; i++) {
        string s; cin >> s;
        if (s == "-") {
            cout << n << '\n';
            continue;
        }
        int cnt = 0;
        for (int j = 0; j < n; j++) {
            if (A[j] == s) cnt++;
        }
        cout << cnt << '\n';
    }
}