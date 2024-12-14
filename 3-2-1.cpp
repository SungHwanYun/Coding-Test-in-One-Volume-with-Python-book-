#include<bits/stdc++.h>
using namespace std;
int main() {
    ios::sync_with_stdio(false);
    cin.tie(NULL);
    cout.tie(NULL);
    string s; int k;
    cin >> s >> k;
    while (s.length() < k) {
        s.push_back(s.back());
    }
    cout << s;
}