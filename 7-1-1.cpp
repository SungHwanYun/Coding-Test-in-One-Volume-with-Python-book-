#include <bits/stdc++.h>
using namespace std;
int main() {
	ios::sync_with_stdio(false);
	cin.tie(NULL);
	cout.tie(NULL);
	int n; cin >> n;
	vector<int> D(n + 1, 1);
	for (int i = 3; i <= n; i++)
		D[i] = (D[i - 1] + D[i - 2]) % 987654321;
	cout << D[n];
}