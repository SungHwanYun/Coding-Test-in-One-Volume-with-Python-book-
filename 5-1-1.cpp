#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
void do_add_query(int T[], int i, int j) {
    T[i]++;
    T[j]--;
}
int translate_time(string t) {
    int x = stoi(t.substr(0, 2)) * 3600 + stoi(t.substr(3, 2)) * 60 + stoi(t.substr(6, 2));
    assert(0 <= x && x < 86400);
    return x;
}
ll get_sum(int T[], int i, int j) {
    for (int t = 1; t < 24 * 60 * 60; t++)
        T[t] += T[t - 1];
    ll ret = 0;
    for (int t = i; t < j; t++)
        ret += T[t];
    return ret;
}
int main() {
    ios::sync_with_stdio(false);
    cin.tie(NULL);
    cout.tie(NULL);

    int T[86400] = { 0 };
    int n; cin >> n;
    while (n--) {
        int op; cin >> op;
        if (op == 1) {
            string x, y; cin >> x >> y;
            do_add_query(T, translate_time(x), translate_time(y));
        }
        else {
            string x, y; cin >> x >> y;
            cout << get_sum(T, translate_time(x), translate_time(y)) << '\n';
        }
    }
}