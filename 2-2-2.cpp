#include<bits/stdc++.h>
using namespace std;
int main() {
    string A; cin >> A;
    string B;
    for (int i = 0; i < A.size(); i++) {
        if ('a' <= A[i] && A[i] <= 'z') B = B + A[i];
    }
    cout << B;
}