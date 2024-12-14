#include<bits/stdc++.h>
using namespace std;
int pow(int x, int y) {
    int ret = 1;
    while (y--) {
        ret *= x;
    }
    return ret;
}
int is_ok(int a) {
    int p = a % 10;
    a /= 10;
    if (p == 0) return 0;
    while (a) {
        int c = a % 10;
        a /= 10;
        if (c == 0 || abs(p - c) > 2) return 0;
        p = c;
    }
    return 1;
}
int main() {
    ios::sync_with_stdio(false);
    cin.tie(NULL);
    cout.tie(NULL);

    int n; cin >> n;
    int answer = 0;
    for (int i = pow(10, n - 1); i < pow(10, n); i++) {
        if (is_ok(i)) answer++;
    }
    cout << answer;
}