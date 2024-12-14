#include<bits/stdc++.h>
using namespace std;
int main() {
    int n, m; cin >> n >> m;
    vector<int>A(n);
    for (auto& a : A)
        cin >> a;

    for (int i = 0; i < m; i++) {
        int op; cin >> op;
        if (op == 1) {
            int x, y, z; cin >> x >> y >> z;
            assert(0 <= x && x <= y && y < n);
            for (int j = x; j <= y; j++)
                A[j] += z;
        }
        else {
            long long answer = 0;
            int x, y; cin >> x >> y;
            assert(0 <= x && x <= y && y < n);
            for (int j = x; j <= y; j++)
                answer += A[j];
            cout << answer << '\n';
        }
    }
}