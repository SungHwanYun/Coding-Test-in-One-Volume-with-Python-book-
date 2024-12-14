#include<bits/stdc++.h>
using namespace std;
int main() {
    vector<string> S;
    do {
        string s; cin >> s;
        S.push_back(s);
    } while (getc(stdin) == ' ');

    map<string, int> D;
    for (auto& s : S) {
        D[s]++;
    }

    vector<pair<string, int>> V;
    for (auto& d : D) {
        V.emplace_back(d.first, d.second);
    }
    sort(V.begin(), V.end());
    for (auto& v : V) {
        cout << v.first << ' ' << v.second << '\n';
    }
}