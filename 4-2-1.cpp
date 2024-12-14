#include<bits/stdc++.h>
using namespace std;
void do_solve(string A, int B) {
    if (A.size() > 0 && A != "0") {
        if (A.back() != '0' || B == 1) {
            cout << A.back();
            B = 1;
        }
        A.pop_back();
        do_solve(A, B);
    }
}
int main() {
    string A; cin >> A;
    do_solve(A, 0);
}