#include <bits/stdc++.h>
using namespace std;

vector<int> answer;
int point = 0;

int get_point(vector<int>& rian, vector<int>& info) {
    int rian_point = 0, apeach_point = 0;

    for (int i = 1; i <= 10; i++) {
        if (rian[i] == 0 && info[i] == 0) continue;

        if (rian[i] > info[i])
            rian_point += i;
        else
            apeach_point += i;
    }
    if (rian_point > apeach_point) return rian_point - apeach_point;
    else return -1;
}

void solve(vector<int>& rian, int n, vector<int>& info) {
    int arrow_sum = 0, remained_arrow;
    for (int i = 0; i < rian.size(); i++)
        arrow_sum += rian[i];
    remained_arrow = n - arrow_sum;

    if (rian.size() == 10) {
        rian.push_back(remained_arrow);
        int x = get_point(rian, info);

        if (x > 0 && x > point) {
            point = x;
            answer = rian;
        }
        rian.pop_back();
        return;
    }

    for (int i = remained_arrow; i >= 0; --i) {
        rian.push_back(i);
        solve(rian, n, info);
        rian.pop_back();
    }
}
vector<int> solution(int n, vector<int> info) {
    vector<int> rian;
    answer.push_back(-1);
    reverse(info.begin(), info.end());
    solve(rian, n, info);
    reverse(answer.begin(), answer.end());
    return answer;
}