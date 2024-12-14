#include <bits/stdc++.h>
#include <unordered_map>
using namespace std;

unordered_map<string, int> mp;
vector<vector<int>> applicant;

vector<int> split_info(string& str) {
    int x, y;

    x = str.find(' ');
    string lang = str.substr(0, x);

    y = str.find(' ', x + 1);
    string job = str.substr(x + 1, y - x - 1);
    x = y;

    y = str.find(' ', x + 1);
    string car = str.substr(x + 1, y - x - 1);
    x = y;

    y = str.find(' ', x + 1);
    string food = str.substr(x + 1, y - x - 1);

    int point = stoi(str.substr(y + 1, str.length() - y - 1));

    vector<int> ret;
    ret.push_back(mp[lang]);
    ret.push_back(mp[job]);
    ret.push_back(mp[car]);
    ret.push_back(mp[food]);
    ret.push_back(point);
    return ret;
}

vector<int> split_query(string& str) {
    int x, y;

    x = str.find(' ');
    string lang = str.substr(0, x);
    x = str.find(' ', x + 1);

    y = str.find(' ', x + 1);
    string job = str.substr(x + 1, y - x - 1);
    x = str.find(' ', y + 1);

    y = str.find(' ', x + 1);
    string car = str.substr(x + 1, y - x - 1);
    x = str.find(' ', y + 1);

    y = str.find(' ', x + 1);
    string food = str.substr(x + 1, y - x - 1);

    int point = stoi(str.substr(y + 1, str.length() - y - 1));

    vector<int> ret;
    ret.push_back(mp[lang]);
    ret.push_back(mp[job]);
    ret.push_back(mp[car]);
    ret.push_back(mp[food]);
    ret.push_back(point);
    return ret;
}

bool is_ok(vector<int>& qry, vector<int>& app) {
    for (int i = 0; i < 4; i++) {
        if (qry[i] != 0 && qry[i] != app[i]) return false;
    }
    return true;
}

vector<int> solution(vector<string> info, vector<string> query) {
    mp["-"] = 0;
    mp["cpp"] = 1; mp["java"] = 2; mp["python"] = 3;
    mp["backend"] = 1; mp["frontend"] = 2;
    mp["junior"] = 1; mp["senior"] = 2;
    mp["chicken"] = 1; mp["pizza"] = 2;

    for (int i = 0; i < info.size(); i++) {
        applicant.push_back(split_info(info[i]));
    }

    vector<int> answer;
    for (int i = 0; i < query.size(); i++) {
        vector<int> v = split_query(query[i]);
        int cnt = 0;
        for (int j = 0; j < applicant.size(); j++) {
            if (is_ok(v, applicant[j]) && v[4] <= applicant[j][4])
                cnt++;
        }
        answer.push_back(cnt);
    }
    return answer;
}