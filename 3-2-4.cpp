#include<bits/stdc++.h>
using namespace std;
void do_add_query(int T[], int i, int j) {
    for (int t = i; t < j; t++)
        T[t]++;
}
int translate_time(string t) {
    int x = stoi(t.substr(0, 2)) * 60 + stoi(t.substr(3, 2));
    assert(0 <= x && x < 3600);
    return x;
}
int main() {
    ios::sync_with_stdio(false);
    cin.tie(NULL);
    cout.tie(NULL);

    int T[3600] = { 0 };
    int n; cin >> n;
    while (n--) {
        int op; cin >> op;
        if (op == 1) {
            string x, y; cin >> x >> y;
            do_add_query(T, translate_time(x), translate_time(y));
        }
        else {
            string x; cin >> x;
            cout << T[translate_time(x)] << '\n';
        }
    }
}