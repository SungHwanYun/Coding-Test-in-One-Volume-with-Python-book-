#include<bits/stdc++.h>
using namespace std;
int main() {
    vector<int> A;
    do {
        int a; cin >> a;
        A.push_back(a);
    } while (getc(stdin) == ' ');

    vector<int> B;
    do {
        int b; cin >> b;
        B.push_back(b);
    } while (getc(stdin) == ' ');

    int a = 0, b = 0;
    for (int i = 0; i < A.size(); i++) {
        if (A[i] > B[i]) a++;
        else if (A[i] < B[i]) b++;
    }
    if (a > b) cout << 1;
    else cout << 0;
}