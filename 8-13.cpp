#include <bits/stdc++.h>
using namespace std;

void add_query(vector<long long>& T, int i, int j) {
    T[i] += 1;
    T[j] -= 1;
}

long long get_max_range(vector<long long>& R, int range_len) {
    long long ret = 0;
    for (int j = range_len - 1; j < 24 * 60 * 60; j++) {
        int i = j - range_len + 1;
        long long a = R[j];
        if (i != 0) {
            a -= R[i - 1];
        }
        ret = max(ret, a);
    }
    return ret;
}

int translate_time(string t) {
    return stoi(t.substr(0, 2)) * 3600 + stoi(t.substr(3, 2)) * 60 + stoi(t.substr(6));
}

long long solution(int n, vector<vector<string>>& A) {
    vector<long long> T(24 * 60 * 60, 0);
    vector<long long> R(24 * 60 * 60, 0);
    long long answer = 0;
    for (auto a : A) {
        if (a[0] == "1") {
            add_query(T, translate_time(a[1]), translate_time(a[2]));
        }
        else {
            for (int t = 1; t < 24 * 60 * 60; t++) {
                T[t] += T[t - 1];
            }
            R[0] = T[0];
            for (int t = 1; t < 24 * 60 * 60; t++) {
                R[t] += R[t - 1] + T[t];
            }
            answer = get_max_range(R, translate_time(a[1]));
        }
    }
    return answer;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    int n;
    cin >> n;
    vector<vector<string>> A(n, vector<string>(3));
    for (int i = 0; i < n; i++) {
        cin >> A[i][0];
        if (A[i][0] == "1") cin >> A[i][1] >> A[i][2];
        else cin >> A[i][1];
    }
    cout << solution(n, A) << endl;
}