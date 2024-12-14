#include <bits/stdc++.h>
#include <unordered_map>
using namespace std;

vector<int> score[3][3][3][4];
unordered_map<string, int> mp;

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
    mp["-"] = 0; mp["cpp"] = 1; mp["java"] = 2; mp["python"] = 3;
    mp["backend"] = 1; mp["frontend"] = 2;
    mp["junior"] = 1; mp["senior"] = 2;
    mp["chicken"] = 1; mp["pizza"] = 2;

    for (int i = 0; i < info.size(); i++) {
        vector<int> v = split_info(info[i]);
        for (int lang = 0; lang <= 3; lang++) {
            for (int job = 0; job <= 2; job++) {
                for (int career = 0; career <= 2; career++) {
                    for (int food = 0; food <= 2; food++) {
                        vector<int> v2 = { lang,job,career,food };
                        if (is_ok(v2, v))
                            score[food][career][job][lang].push_back(v[4]);
                    }
                }
            }
        }
    }

    for (int lang = 0; lang <= 3; lang++) {
        for (int job = 0; job <= 2; job++) {
            for (int career = 0; career <= 2; career++) {
                for (int food = 0; food <= 2; food++) {
                    sort(score[food][career][job][lang].begin(), score[food][career][job][lang].end());
                }
            }
        }
    }

    vector<int> answer;
    for (int i = 0; i < query.size(); i++) {
        vector<int> v = split_query(query[i]);
        int lang = v[0], job = v[1], car = v[2], food = v[3], point = v[4];
        int len = score[food][car][job][lang].end() - lower_bound(score[food][car][job][lang].begin(), score[food][car][job][lang].end(), point);
        answer.push_back(len);
    }
    return answer;
}