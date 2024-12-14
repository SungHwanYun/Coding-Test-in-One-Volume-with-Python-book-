#include <bits/stdc++.h>
#include <unordered_map>
using namespace std;

int translate_days(string x) {
    return stoi(x.substr(0, 4)) * 12 * 28 + stoi(x.substr(5, 2)) * 28 + stoi(x.substr(8));
}

vector<int> solution(string today, vector<string> terms, vector<string> privacies) {
    unordered_map<string, int> T;
    for (string t : terms) {
        string x, y;
        istringstream iss(t);
        iss >> x >> y;
        T[x] = stoi(y) * 28;
    }
    vector<int> answer;
    int today_days = translate_days(today);
    for (int i = 0; i < privacies.size(); i++) {
        string x, y;
        istringstream iss(privacies[i]);
        iss >> x >> y;
        int a = translate_days(x) + T[y];
        if (a <= today_days) {
            answer.push_back(i + 1);
        }
    }
    return answer;
}