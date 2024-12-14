#include<bits/stdc++.h>
using namespace std;
int main() {
    string A; cin >> A;
    string B;
    for (int i = 0; i < A.size(); i++) {
        char b;
        if ('A' <= A[i] && A[i] <= 'Z') b = A[i];
        else b = A[i] + 'A' - 'a';
        B = B + b;
    }
    cout << B;
}