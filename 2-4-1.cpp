#include<bits/stdc++.h>
using namespace std;
typedef pair<int, int> pii;
int main() {
    pii answer = { 0, 0 };
    int n; cin >> n;
    deque<int> q;
    for (int i = 0; i < n; i++) {
        int op; cin >> op;
        if (op == 1) {
            int a; cin >> a;
            q.push_back(a);
            if (answer.first < q.size() || (answer.first == q.size() && answer.second > q.back()))
                answer = { q.size(), q.back() };
        }
        else {
            q.pop_front();
        }
    }
    cout << answer.first << ' ' << answer.second;
}