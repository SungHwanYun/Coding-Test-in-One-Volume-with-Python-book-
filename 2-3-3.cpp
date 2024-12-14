#include<bits/stdc++.h>
using namespace std;
int main() {
    vector<string> A;
    do {
        string s; cin >> s;
        A.push_back(s);
    } while (getc(stdin) == ' ');

    vector<string> B;
    do {
        string s; cin >> s;
        B.push_back(s);
    } while (getc(stdin) == ' ');

    map<string, int> D;
    for (auto& b : B) {
        D[b]++;
    }

    vector<string> answer;
    for (auto& a : A) {
        if (D[a] == 0) {
            answer.push_back(a);
        }
    }
    sort(answer.begin(), answer.end());
    for (auto& a : answer) {
        cout << a << '\n';
    }
}