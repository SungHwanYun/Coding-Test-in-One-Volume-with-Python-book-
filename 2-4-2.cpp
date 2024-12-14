#include<bits/stdc++.h>
using namespace std;
typedef pair<int, int> pii;
int main() {
    vector<int> answer[3];
    int n; cin >> n;
    deque<pii> q;
    for (int i = 0; i < n; i++) {
        int op; cin >> op;
        if (op == 1) {
            int a, b; cin >> a >> b;
            q.push_back({ a, b });
        }
        else {
            int a = q.front().first, b = q.front().second;
            q.pop_front();
            int bb; cin >> bb;
            if (b == bb) {
                answer[0].push_back(a);
            }
            else {
                answer[1].push_back(a);
            }
        }
    }

    while (q.size() > 0) {
        int a = q.front().first; q.pop_front();
        answer[2].push_back(a);
    }

    for (int i = 0; i < 3; i++) {
        sort(answer[i].begin(), answer[i].end());
        if (answer[i].size() == 0) {
            cout << "None\n";
        }
        else {
            for (int j = 0; j < answer[i].size(); j++) {
                cout << answer[i][j];
                if (j < answer[i].size() - 1) cout << ' ';
                else cout << '\n';
            }
        }
    }
}