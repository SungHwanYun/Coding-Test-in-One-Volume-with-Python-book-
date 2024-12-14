#include<bits/stdc++.h>
using namespace std;
int main() {
    int n;
    cin >> n;
    vector<int> A(n);
    for (int i = 0; i < n; i++) cin >> A[i];

    int i, j, k;
    cin >> i >> j >> k;
    for (int idx = i; idx <= j; idx++) A[idx] *= k;

    int answer = 0;
    for (int i = 0; i < n; i++) answer += A[i];
    cout << answer;
}