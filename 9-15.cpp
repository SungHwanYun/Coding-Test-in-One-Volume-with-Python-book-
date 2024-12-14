#include <bits/stdc++.h>
using namespace std;

string solution(string new_id) {
    int i;
    string answer = "";

    for (i = 0; i < new_id.length(); i++) {
        if ('A' <= new_id[i] && new_id[i] <= 'Z')
            new_id[i] = 'a' + new_id[i] - 'A';
    }

    for (i = 0; i < new_id.length(); i++) {
        if (('a' <= new_id[i] && new_id[i] <= 'z') || ('0' <= new_id[i] && new_id[i] <= '9') || new_id[i] == '-' || new_id[i] == '_' ||
            new_id[i] == '.') {
            answer += new_id[i];
        }
    }
    new_id = answer;
    answer = "";

    if (new_id.length() > 0)
        answer += new_id[0];
    for (i = 1; i < new_id.length(); i++) {
        if (new_id[i - 1] != '.' || new_id[i] != '.') {
            answer += new_id[i];
        }
    }
    new_id = answer;
    answer = "";

    if (new_id.length() > 0 && new_id[0] != '.')
        answer += new_id[0];
    for (i = 1; i < new_id.length() - 1; i++)
        answer += new_id[i];
    if (new_id.length() > 1 && new_id[new_id.length() - 1] != '.')
        answer += new_id[new_id.length() - 1];
    new_id = answer;

    if (new_id.length() == 0)
        new_id += 'a';

    if (new_id.length() > 15) {
        new_id = new_id.substr(0, 15);
        if (new_id[14] == '.')
            new_id = new_id.substr(0, 14);
    }

    while (new_id.length() <= 2)
        new_id += new_id[new_id.length() - 1];

    answer = new_id;
    return answer;
}