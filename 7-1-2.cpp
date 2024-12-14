#include <bits/stdc++.h>
using namespace std;
int main(int argc, char* argv[]) {
	ios::sync_with_stdio(false);
	cin.tie(NULL);
	cout.tie(NULL);
	int a, k; cin >> a >> k;
	vector<int> D(k + 1, 0);
	for (int i = a + 1; i <= k; i++) {
		D[i] = D[i - 1] + 1;
		if (i % 2 == 0 && i / 2 >= a) {
			D[i] = min(D[i], D[i / 2] + 1);
		}
	}
	cout << D[k];
}