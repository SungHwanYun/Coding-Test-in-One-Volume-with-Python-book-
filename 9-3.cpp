#include <bits/stdc++.h>
using namespace std;

vector<int> get_info(vector<int>& user, vector<int>& emoticons, vector<int>& x) {
    int m = emoticons.size();
    int money = 0;
    for (int i = 0; i < m; i++) {
        if (x[i] >= user[0]) {
            money += emoticons[i] * (100 - x[i]) / 100;
        }
    }
    if (money >= user[1]) {
        return { 1, 0 };
    }
    else {
        return { 0, money };
    }
}

vector<int> solution(vector<vector<int>> users, vector<int> emoticons) {
    int n = users.size();
    int m = emoticons.size();
    vector<int> answer = { 0, 0 };
    for (int k = 0; k < (1<<(2*m)); k++) {
        vector<int> x;
        for (int i = 0; i < m; i++) {
            int a = (k >> (i * 2)) & 0x3;
            x.push_back((a + 1) * 10);
        }
        vector<int> ans = { 0, 0 };
        for (auto u : users) {
            vector<int> ret = get_info(u, emoticons, x);
            ans[0] += ret[0];
            ans[1] += ret[1];
        }
        if (answer[0] < ans[0] || (answer[0] == ans[0] && answer[1] < ans[1])) {
            answer[0] = ans[0];
            answer[1] = ans[1];
        }
    }
    return answer;
}
