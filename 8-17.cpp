#include <bits/stdc++.h>
using namespace std;

int solution(vector<vector<int>>& A) {
    vector<int> src = { 0, 1, 2, 3, 4, 5 };
    int answer = (int)1e8;
    do {
        vector<vector<int>> D(6, vector<int>(2, 0));
        D[0][0] = D[0][1] = A[src[0] * 2][src[0] * 2 + 1];
        for (int i = 1; i < 6; i++) {
            D[i][0] = min(D[i - 1][0] + A[src[i - 1] * 2][src[i] * 2 + 1] + A[src[i] * 2][src[i] * 2 + 1],
                D[i - 1][1] + A[src[i - 1] * 2 + 1][src[i] * 2 + 1] + A[src[i] * 2][src[i] * 2 + 1]);
            D[i][1] = min(D[i - 1][0] + A[src[i - 1] * 2][src[i] * 2] + A[src[i] * 2][src[i] * 2 + 1],
                D[i - 1][1] + A[src[i - 1] * 2 + 1][src[i] * 2] + A[src[i] * 2][src[i] * 2 + 1]);
        }
        answer = min(answer, min(D[5][0], D[5][1]));
    } while (next_permutation(src.begin(), src.end()));
    return answer;
}

int main() {
    vector<vector<int>> A(12, vector<int>(12));
    for (int i = 0; i < 12; i++) {
        for (int j = 0; j < 12; j++) {
            cin >> A[i][j];
        }
    }
    cout << solution(A) << endl;
}