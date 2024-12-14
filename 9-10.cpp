#include <bits/stdc++.h>
using namespace std;

int parking_time[10000];
int in_time[10000];

int get_num(string& s, int mode) {
    if (mode == 0) {
        int h = (s[0] - '0') * 10 + (s[1] - '0');
        int m = (s[3] - '0') * 10 + (s[4] - '0');
        return h * 60 + m;
    }
    else if (mode == 1) {
        return (s[6] - '0') * 1000 + (s[7] - '0') * 100 + (s[8] - '0') * 10 + (s[9] - '0');
    }
    else if (mode == 2) {
        if (s[11] == 'I') return 0;
        return 1;
    }
}

int get_fee(vector<int> fees, int t) {
    int money = fees[1];
    if (fees[0] < t)
        money += (t - fees[0] + fees[2] - 1) / fees[2] * fees[3];
    return money;
}

vector<int> solution(vector<int> fees, vector<string> records) {
    vector<int> answer;
    for (int i = 0; i < 10000; i++)
        in_time[i] = -1;

    for (int i = 0; i < records.size(); i++) {
        int t = get_num(records[i], 0);
        int c = get_num(records[i], 1);
        int d = get_num(records[i], 2);

        if (d == 0) {
            in_time[c] = t;
        }
        else {
            parking_time[c] += t - in_time[c];
            in_time[c] = -1;
        }
    }

    for (int i = 0; i < 10000; i++) {
        if (in_time[i] != -1) {
            parking_time[i] += 23 * 60 + 59 - in_time[i];
        }
    }

    for (int i = 0; i < 10000; i++) {
        if (parking_time[i] != 0) {
            answer.push_back(get_fee(fees, parking_time[i]));
        }
    }

    return answer;
}