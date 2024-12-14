#include <bits/stdc++.h>
using namespace std;

string solution(int n, int k) {
    string a = "";
    while (n > 0) {
        int d = n % k;
        n = n / k;
        a += to_string(d);
    }
    reverse(a.begin(), a.end());
    a += "0";
    long long c = 0;
    size_t pos = 0;
    size_t prev_pos = 0;
    while ((pos = a.find('0', pos)) != string::npos) {
        string b = a.substr(prev_pos, pos - prev_pos);
        if (!b.empty()) {
            c += stoll(b);
        }
        pos++;
        prev_pos = pos;
    }
    string ret = "";
    while (c > 0) {
        int d = c % k;
        c = c / k;
        ret += to_string(d);
    }
    reverse(ret.begin(), ret.end());
    return ret;
}

int main() {
    int n, k;
    cin >> n >> k;
    cout << solution(n, k) << endl;
    return 0;
}