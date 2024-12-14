#include<bits/stdc++.h>
using namespace std;
int parse_log(string& s) {
    string hour = s.substr(0, 2);
    string minute = s.substr(3, 2);
    return stoi(hour) * 60 + stoi(minute);
}
int main() {
    vector<string> A;
    do {
        string a; cin >> a;
        A.push_back(a);
    } while (getc(stdin) == ' ');

    int total_time = 0;
    for (auto& a : A) {
        int t = parse_log(a);
        total_time += t;
    }

    int hour = total_time / 60;
    int minute = total_time % 60;

    char buf[100];
    if (hour < 100) sprintf(buf, "%02d:%02d", hour, minute);
    else sprintf(buf, "%d:%02d", hour, minute);
    cout << buf;
}