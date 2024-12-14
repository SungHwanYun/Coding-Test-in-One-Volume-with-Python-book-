#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
int main() {
    ios::sync_with_stdio(false);
    cin.tie(NULL);
    cout.tie(NULL);

    int n, k;
    cin >> n >> k;

    ll b = 0;
    while (n > 0) {
        int d = n % k;
        n /= k;
        b = b * k + d;
    }
    cout << b;
}