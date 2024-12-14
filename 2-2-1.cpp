#include<bits/stdc++.h>
using namespace std;
int main() {
    string A; cin >> A;
    string B;
    for (int i = 1; i < A.size(); i += 2) {
        B = B + A[i];
    }
    cout << B;
}