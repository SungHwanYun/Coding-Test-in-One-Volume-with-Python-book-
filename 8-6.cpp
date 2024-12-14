#include <bits/stdc++.h>
using namespace std;

int solve(int n, int k, int a, vector<int>& B) {
    if (a == n) {
        return 0;
    }
    for (int b = a + 1; b <= a + k; b++) {
        if (b > n) {
            break;
        }
        if (B[b] == 1) {
            continue;
        }
        if (solve(n, k, b, B) == 0) {
            return 1;
        }
    }
    return 0;
}

int solution(int n, int k, vector<int>& A) {
    vector<int> B(n + 1, 0);
    for (int a : A) {
        B[a] = 1;
    }
    return solve(n, k, 0, B);
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int n, k;
    scanf("%d %d\n", &n, &k);
    vector<int> A;
    do {
        int a; scanf("%d", &a); A.push_back(a);
    } while (getc(stdin) == ' ');
    cout << solution(n, k, A) << endl;
}