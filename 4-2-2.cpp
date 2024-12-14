#include<bits/stdc++.h>
using namespace std;
int do_solve(int n) {
    if (n == 1) return 1;
    return n + do_solve(n - 1);
}
int main() {
    int n; cin >> n;
    cout << do_solve(n);
}