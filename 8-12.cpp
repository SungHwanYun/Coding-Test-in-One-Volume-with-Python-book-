#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    int n;
    cin >> n;
    vector<vector<string>> A(n, vector<string>(3));
    for (int i = 0; i < n; i++) {
        cin >> A[i][0] >> A[i][1] >> A[i][2];
    }
    vector<long long> T(24 * 60 * 60, 0);
    vector<long long> R(24 * 60 * 60, 0);
    vector<long long> answer;
    bool flag = false;
    for (int i = 0; i < n; i++) {
        if (A[i][0] == "1") {
            int start = stoi(A[i][1].substr(0, 2)) * 3600 + stoi(A[i][1].substr(3, 2)) * 60 + stoi(A[i][1].substr(6));
            int end = stoi(A[i][2].substr(0, 2)) * 3600 + stoi(A[i][2].substr(3, 2)) * 60 + stoi(A[i][2].substr(6));
            T[start]++; T[end]--;
        }
        else {
            if (!flag) {
                for (int j = 1; j < 24 * 60 * 60; j++) {
                    T[j] += T[j - 1];
                }
                flag = true;
                R[0] = T[0];
                for (int j = 1; j < 24 * 60 * 60; j++) {
                    R[j] += R[j - 1] + T[j];
                }
            }
            int start = stoi(A[i][1].substr(0, 2)) * 3600 + stoi(A[i][1].substr(3, 2)) * 60 + stoi(A[i][1].substr(6));
            int end = stoi(A[i][2].substr(0, 2)) * 3600 + stoi(A[i][2].substr(3, 2)) * 60 + stoi(A[i][2].substr(6));
            long long ret = R[end - 1];
            if (start != 0) {
                ret -= R[start - 1];
            }
            answer.push_back(ret);
        }
    }
    for (int i = 0; i < answer.size(); i++) {
        cout << answer[i] << "\n";
    }
    return 0;
}