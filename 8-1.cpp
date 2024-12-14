#include <bits/stdc++.h>
using namespace std;

vector<string> split(string s, char delimiter) {
    vector<string> tokens;
    string token;
    for (char c : s) {
        if (c == delimiter) {
            tokens.push_back(token);
            token = "";
        }
        else {
            token += c;
        }
    }
    tokens.push_back(token);
    return tokens;
}

pair<int, string> parse_log(string s) {
    vector<string> tokens = split(s, ' ');
    int t = stoi(tokens[0].substr(0, 2)) * 60 + stoi(tokens[0].substr(3, 2));
    string name = tokens[1];
    return make_pair(t, name);
}

int get_fee(vector<int> fees, int t) {
    int money = fees[1];
    if (fees[0] < t) {
        money += (t - fees[0] + fees[2] - 1) / fees[2] * fees[3];
    }
    return money;
}

vector<pair<string, int>> solution(int n, vector<string> A, vector<int> fees) {
    map<string, int> d;
    for (auto log : A) {
        pair<int, string> parsed_log = parse_log(log);
        int t = parsed_log.first;
        string name = parsed_log.second;
        if (d.find(name) != d.end()) {
            d[name] += t;
        }
        else {
            d[name] = t;
        }
    }
    for (auto& entry : d) {
        entry.second = get_fee(fees, entry.second);
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
    return answer;
}

int main() {
    int n;
    scanf("%d\n", &n);
    vector<string> A(n);
    for (int i = 0; i < n; i++) {
        getline(cin, A[i]);
    }
    vector<int> fees = { 100, 10, 50, 3 };
    vector<pair<string, int>> B = solution(n, A, fees);
    for (auto entry : B) {
        cout << entry.first << " " << entry.second << endl;
    }
    return 0;
}
