#include<bits/stdc++.h>
using namespace std;
int main() {
    ios::sync_with_stdio(false);
    cin.tie(NULL);
    cout.tie(NULL);
    string A; cin >> A;
    sort(A.begin(), A.end());
    do {
        cout << A << '\n';
    } while (next_permutation(A.begin(), A.end()));
}