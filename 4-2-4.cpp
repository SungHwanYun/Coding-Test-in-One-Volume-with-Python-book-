#include<bits/stdc++.h>
using namespace std;
int do_solve(vector<int>& A, int n) {
    if (A.size() == n) return 1;
    int s, e;
    if (A.size() == 0) {
        s = 1; e = 9;
    }
    else {
        s = max(A.back() - 2, 1);
        e = min(A.back() + 2, 9);
    }

    int ret = 0;
    for (int i = s; i <= e; i++) {
        A.push_back(i);
        ret += do_solve(A, n);
        A.pop_back();
    }
    return ret;
}
int main() {
    int n; cin >> n;
    vector<int> A;
    cout << do_solve(A, n);
}