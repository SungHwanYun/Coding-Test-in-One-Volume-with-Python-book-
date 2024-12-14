#include<bits/stdc++.h>
using namespace std;
int translate_time(string t) {
    int x = stoi(t.substr(0, 2)) * 60 + stoi(t.substr(3, 2));
    assert(0 <= x && x < 3600);
    return x;
}
int get_fee(int fees[], int t) {
    int money = fees[1];
    if (fees[0] < t) {
        money += (t - fees[0] + fees[2] - 1) / fees[2] * fees[3];
    }
    return money;
}
int main() {
    int fees[4] = { 100, 10, 50, 3 };
    int total_cost = 0;
    do {
        string t; cin >> t;
        total_cost += get_fee(fees, translate_time(t));
    } while (getc(stdin) == ' ');
    cout << total_cost;
}