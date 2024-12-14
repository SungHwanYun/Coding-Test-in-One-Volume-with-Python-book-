#include <bits/stdc++.h>
#include <unordered_map>
using namespace std;

int N;
int report_result[1004][1004];
int bad_id[1004];

int get_id_num_slow(vector<string>& id_list, string& s) {
    for (int i = 0; i < id_list.size(); i++) {
        if (id_list[i] == s) return i;
    }
    return -1;
}

int get_id_num_fast(unordered_map<string, int>& mp, string& s) {
    return mp[s];
}

vector<int> solution(vector<string> id_list, vector<string> report, int k) {
    unordered_map<string, int> mp;
    for (int i = 0; i < id_list.size(); i++)
        mp[id_list[i]] = i;

    N = id_list.size();
    for (int i = 0; i < report.size(); i++) {
        size_t space_pos = report[i].find(' ');

        string x = report[i].substr(0, space_pos);

        string y = report[i].substr(space_pos + 1, report[i].length() - space_pos - 1);

        int xidx = get_id_num_fast(mp, x);
        int yidx = get_id_num_fast(mp, y);

        report_result[xidx][yidx] = 1;
    }

    for (int j = 0; j < N; j++) {
        int cnt = 0;
        for (int i = 0; i < N; i++)
            cnt += report_result[i][j];
        if (cnt >= k)
            bad_id[j] = 1;
        else
            bad_id[j] = 0;
    }

    vector<int> answer;
    for (int i = 0; i < N; i++) {
        int cnt = 0;
        for (int j = 0; j < N; j++) {
            if (report_result[i][j] == 1 && bad_id[j] == 1)
                cnt++;
        }
        answer.push_back(cnt);
    }
    return answer;
}