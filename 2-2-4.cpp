#include<bits/stdc++.h>
using namespace std;
int main() {
    string A; cin >> A;
    vector<char> B;
    do {
        char b; cin >> b;
        B.push_back(b);
    } while (getc(stdin) == ' ');

    for (int i = 0; i < B.size(); i++) {
        char b = B[i];
        for (int j = 0; j < A.size(); j++) {
            if (A[j] == b) A[j] = b + 'a' - 'A';
        }
    }
    cout << A;
}