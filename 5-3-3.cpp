#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
int is_prime(ll a) {
    if (a < 2) return 0;
    for (ll i = 2; i * i <= a; i++) {
        if (a % i == 0) return 0;
    }
    return 1;
}
int main() {
    ll answer = 0;
    do {
        ll a; cin >> a;
        if (is_prime(a) == 1)
            answer += a;
    } while (getc(stdin) == ' ');
    cout << answer;
}