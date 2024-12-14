#include<bits/stdc++.h>
using namespace std;
int main() {
    ios::sync_with_stdio(false);
    cin.tie(NULL);
    cout.tie(NULL);

    string A; int k, n;
    cin >> A >> k;
    n = A.size();

    vector<int> perm(n, 0);
    for (int i = n - k; i < n; i++) perm[i] = 1;

    vector<string> C;
    do {
        string s = "";
        for (int i = 0; i < n; i++) {
            if (perm[i] == 1)
                s.push_back(A[i]);
        }
        C.push_back(s);
    } while (next_permutation(perm.begin(), perm.end()));

    sort(C.begin(), C.end());
    for (auto& c : C) cout << c << '\n';
}