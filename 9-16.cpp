#include <bits/stdc++.h>
#include <unordered_map>
using namespace std;

unordered_map<string, int> mp[11];

void build_menu(string& src, int idx, int cnt, string& dst) {
    int n = src.length();
    int remain = n - idx;

    if (remain < cnt) return;

    if (idx == n || cnt == 0) {
        if (cnt == 0) {
            if (mp[dst.length()].find(dst) == mp[dst.length()].end())
                mp[dst.length()][dst] = 1;
            else
                mp[dst.length()][dst]++;
        }
        return;
    }

    build_menu(src, idx + 1, cnt, dst);

    dst.push_back(src[idx]);
    build_menu(src, idx + 1, cnt - 1, dst);
    dst.pop_back();
}

vector<string> solution(vector<string> orders, vector<int> course) {
    vector<string> answer;

    for (int i = 0; i < orders.size(); i++)
        sort(orders[i].begin(), orders[i].end());

    for (int i = 0; i < orders.size(); i++) {
        for (int j = 0; j < course.size(); j++) {
            string s="";
            build_menu(orders[i], 0, course[j], s);
        }
    }

    for (int i = 0; i < course.size(); i++) {
        int len = course[i];
        int mx = -1;
        for (auto it = mp[len].begin(); it != mp[len].end(); ++it) {
            mx = max(mx, it->second);
        }

        if (mx < 2) continue;

        for (auto it = mp[len].begin(); it != mp[len].end(); ++it) {
            if (it->second == mx)
                answer.push_back(it->first);
        }
    }

    sort(answer.begin(), answer.end());
    return answer;
}