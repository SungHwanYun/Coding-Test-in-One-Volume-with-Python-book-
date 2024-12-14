#include <bits/stdc++.h>
using namespace std;

int solution(int n, int k, vector<int>& A) {
    vector<int> B(n + 1, 0);
    for (int a : A) {
        B[a] = 1;
    }
    vector<int> D(n + 2, 0);
    for (int i = n; i > 0; i--) {
        for (int j = i; j <= i + k - 1; j++) {
            if (j > n) break;
            if (B[j] == 1) {
                continue;
            }
            if (D[j + 1] == 0) {
                D[i] = 1;
                break;
            }
        }
    }
    return D[1];
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