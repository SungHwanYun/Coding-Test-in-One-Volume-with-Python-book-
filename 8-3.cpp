#include <bits/stdc++.h>
using namespace std;

bool is_ok(vector<string>& qry, vector<string>& student) {
    for (int i = 0; i < 3; i++) {
        if (qry[i] != "-" && qry[i] != student[i]) {
            return false;
        }
    }
    return true;
}

vector<int> solution(int n, int m, vector<vector<string>>& A, vector<vector<string>>& B) {
    vector<int> answer;
    for (auto qry : B) {
        int cnt = 0;
        for (auto student : A) {
            if (is_ok(qry, student)) {
                cnt += 1;
            }
        }
        answer.push_back(cnt);
    }
    return answer;
}

int main() {
    int n, m;
    cin >> n >> m;
    vector<vector<string>> A(n, vector<string>(3));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < 3; j++) {
            cin >> A[i][j];
        }
    }
    vector<vector<string>> B(m, vector<string>(3));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < 3; j++) {
            cin >> B[i][j];
        }
    }
    vector<int> C = solution(n, m, A, B);
    for (auto c : C) {
        cout << c << endl;
    }
    return 0;
}