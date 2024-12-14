#include<bits/stdc++.h>
using namespace std;
typedef pair<int, int> pii;
int do_solve(int n) {
    if (n <= 2) return 1;
    return do_solve(n - 1) + do_solve(n - 2);
}
int main() {
    int n; cin >> n;
    cout << do_solve(n);
}