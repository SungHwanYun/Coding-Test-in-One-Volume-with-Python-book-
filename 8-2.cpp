#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    cin >> n;
    cin.ignore();

    string A;
    getline(cin, A);

    vector<string> B(n);
    for (int i = 0; i < n; i++) {
        getline(cin, B[i]);
    }

    map<string, int> d;
    string temp;
    for (int i = 0; i < A.size(); i++) {
        if (A[i] == ' ') {
            d[temp] = 0;
            temp = "";
        }
        else {
            temp += A[i];
        }
    }
    d[temp] = 0;

    for (int i = 0; i < n; i++) {
        temp = "";
        for (int j = 0; j < B[i].size(); j++) {
            if (B[i][j] == ' ') {
                d[temp]++;
                temp = "";
            }
            else {
                temp += B[i][j];
            }
        }
        d[temp]++;
    }

    vector<pair<string, int>> answer(d.begin(), d.end());
    sort(answer.begin(), answer.end(), [](const pair<string, int>& a, const pair<string, int>& b) {
        if (a.second != b.second) {
            return a.second > b.second;
        }
        else {
            return a.first < b.first;
        }
        });

    for (const auto& p : answer) {
        cout << p.first << " " << p.second << endl;
    }
}
