#include<bits/stdc++.h>
using namespace std;
int main() {
    int n, k;
    cin >> n >> k;
    vector<int> A(n);
    for (int i = 0; i < n; i++) cin >> A[i];

    int answer = 0;
    for (int i = 0; i < n; i++) {
        if (A[i] == k) answer++;
    }
    cout << answer;
}