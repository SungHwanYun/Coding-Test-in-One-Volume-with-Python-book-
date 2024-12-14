#include <bits/stdc++.h>
using namespace std;

int solve(int n, int k, int a, vector<int>& B) {
    if (a == n) {
        return 0;
    }
    vector<int> nxt;
    for (int b = a + 1; b <= a + k; b++) {
        if (b > n) {
            break;
        }
        if (B[b] == 1) {
            continue;
        }
        nxt.push_back(solve(n, k, b, B));
    }
    if (nxt.size() == 0) {
        return 0;
    }
    sort(nxt.begin(), nxt.end());
    if (nxt[0] > 0) {
        return -(nxt[nxt.size() - 1] + 1);
    }
    int ret = 0;
    for (int i = 0; i < nxt.size(); i++) {
        if (nxt[i] <= 0) {
            ret = nxt[i];
        }
    }
    return -ret + 1;
}

int solution(int n, int k, vector<int>& A) {
    vector<int> B(n + 1, 0);
    for (int a : A) {
        B[a] = 1;
    }
    return abs(solve(n, k, 0, B));
}

int main() {
    int n, k;
    scanf("%d %d\n", &n, &k);
    vector<int> A;
    do {
        int a; scanf("%d", &a); A.push_back(a);
    } while (getc(stdin) == ' ');
    cout << solution(n, k, A) << endl;
    return 0;
}