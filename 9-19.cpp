#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int A[360004];
ll B[360004];
int P, T;

int convert_time(const string& str) {
    int h = stoi(str.substr(0, 2)), m = stoi(str.substr(3, 2)), s = stoi(str.substr(6, 2));
    return h * 3600 + m * 60 + s;
}

string solution(string play_time, string adv_time, vector<string> logs) {
    P = convert_time(play_time);
    T = convert_time(adv_time);

    for (int i = 0; i < logs.size(); i++) {
        int s = convert_time(logs[i]);
        int e = convert_time(logs[i].substr(9, 8));
        A[s]++; A[e]--;
    }

    for (int i = 1; i <= P; i++)
        A[i] += A[i - 1];

    B[0] = A[0];
    for (int i = 1; i <= P; i++)
        B[i] = B[i - 1] + A[i];

    int x = 0;
    ll y = B[T - 1];
    for (int i = 1; i + T <= P; i++) {
        ll sum = B[i + T - 1] - B[i - 1];
        if (sum > y) {
            y = sum;
            x = i;
        }
    }

    char str[20];
    int h = x / 3600;
    int m = (x - h * 3600) / 60;
    int s = x % 60;
    sprintf(str, "%02d:%02d:%02d", h, m, s);
    string answer = str;
    return answer;
}