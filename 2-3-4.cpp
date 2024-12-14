#include<bits/stdc++.h>
using namespace std;
int main() {
    vector<string> A;
    do {
        string s; cin >> s;
        A.push_back(s);
    } while (getc(stdin) == ' ');

    string B; cin >> B;
    map<string, int> D;
    for (auto& a : A) {
        for (int i = 1; i < a.size(); i++) {
            string s = a.substr(0, i);
            D[s]++;
        }
    }
    cout << D[B];
}