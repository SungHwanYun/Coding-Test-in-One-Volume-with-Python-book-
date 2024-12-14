#include<bits/stdc++.h>
using namespace std;
int main() {
    int n, m; cin >> n >> m;
    vector<pair<string, int>> A;
    for (int i = 0; i < n; i++) {
        string s; int cost;
        cin >> s >> cost;
        A.emplace_back(s, cost);
    }
    vector<string> B;
    for (int i = 0; i < m; i++) {
        string s; cin >> s;
        B.push_back(s);
    }

    map<string, int> D;
    for (auto& a : A) {
        D[a.first] = a.second;
    }

    long long answer = 0;
    for (auto& b : B) {
        answer += D[b];
    }
    cout << answer;
}