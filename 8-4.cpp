#include <bits/stdc++.h>
#include <unordered_map>
using namespace std;

void solve(vector<string>& C, unordered_map<string, int>& d) {
    for (string c : C) {
        if (d.find(c) != d.end()) {
            d[c]++;
        }
        else {
            d[c] = 1;
        }
    }
}

vector<string> get_combinations(string& s, int r) {
    int n = s.length();
    vector<bool> v(n);
    fill(v.begin(), v.begin() + r, true);
    vector<string> ans;
    do {
        string x = "";
        for (int i = 0; i < n; ++i) {
            if (v[i]) {
                x += s[i];
            }
        }
        ans.push_back(x);
    } while (std::prev_permutation(v.begin(), v.end()));
    return ans;
}

vector<string> solution(string X, string Y, string Z, int k) {
    vector<string> CX, CY, CZ;
    CX = get_combinations(X, k);
    CY = get_combinations(Y, k);
    CZ = get_combinations(Z, k);
    unordered_map<string, int> d;
    solve(CX, d);
    solve(CY, d);
    solve(CZ, d);
    vector<string> answer;
    for (auto it = d.begin(); it != d.end(); it++) {
        if (it->second >= 2) {
            answer.push_back(it->first);
        }
    }
    sort(answer.begin(), answer.end());
    if (answer.size() == 0) {
        answer.push_back("-1");
    }
    return answer;
}

int main() {
    string X, Y, Z;
    cin >> X >> Y >> Z;
    int k;
    cin >> k;
    vector<string> C = solution(X, Y, Z, k);
    for (string c : C) {
        cout << c << endl;
    }
    return 0;
}