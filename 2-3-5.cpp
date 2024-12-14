#include<bits/stdc++.h>
using namespace std;
int main() {
    int n; cin >> n;
    vector<int> A;
    map<int, int>D;
    for (int i = 0; i < n; i++) {
        int a; cin >> a;
        A.push_back(a);
        D[a]++;
    }

    int mx = -1;
    for (auto& d : D) {
        mx = max(mx, d.second);
    }

    vector<int>answer;
    for (auto& d : D) {
        if (d.second == mx)
            answer.push_back(d.first);
    }
    sort(answer.begin(), answer.end());
    for (int i = 0; i < answer.size(); i++) {
        cout << answer[i];
        if (i < answer.size() - 1) cout << ' ';
    }
}