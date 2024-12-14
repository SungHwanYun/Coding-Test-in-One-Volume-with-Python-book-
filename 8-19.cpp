#include <bits/stdc++.h>
using namespace std;

int solution(int n, int k, vector<int>& A) {
    vector<int> B(n + 1, 0);
    for (int a : A) {
        B[a] = 1;
    }
    vector<int> D(n + 2, 0);
    for (int i = n; i > 0; i--) {
        vector<int> nxt;
        for (int j = i; j <= i + k - 1; j++) {
            if (j > n) break;
            if (B[j] == 1) continue;
            nxt.push_back(D[j + 1]);
        }
        if (nxt.size() == 0) {
            D[i] = 0;
            continue;
        }
        sort(nxt.begin(), nxt.end());
        if (nxt[0] > 0) {
            D[i] = -(nxt[nxt.size() - 1] + 1);
            continue;
        }
        int ret = 0;
        for (int p = 0; p < nxt.size(); p++) {
            if (nxt[p] <= 0) {
                ret = nxt[p];
            }
        }
        D[i] = -ret + 1;
    }
    return abs(D[1]);
}

int main() {
    int n, k;
    cin >> n >> k;
    vector<int> A;
    do {
        int a; cin >> a;
        A.push_back(a);
    } while (getc(stdin) != '\n');
    cout << solution(n, k, A) << endl;
}