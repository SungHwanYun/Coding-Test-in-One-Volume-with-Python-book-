#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
int main() {
    ios::sync_with_stdio(false);
    cin.tie(NULL);
    cout.tie(NULL);

    int n, k;
    cin >> n >> k;

    ll a = 0;
    while (n > 0) {
        int d = n % k;
        n /= k;
        a = a + d;
    }

    string b = "";
    while (a > 0) {
        int d = a % k;
        a = a / k;
        b = to_string(d) + b;
    }
    cout << b;
}