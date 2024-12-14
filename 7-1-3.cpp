#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef pair<int, int> pii;

int main(int argc, char* argv[]) {
	ios::sync_with_stdio(false);
	cin.tie(NULL);
	cout.tie(NULL);

	int n; cin >> n;
	vector<vector<int>> D(n + 1);
	for (int i = 0; i <= n; i++)
		D[i] = vector<int>(10, 0);
	for (int j = 1; j <= 9; j++)
		D[1][j] = 1;
	for (int i = 2; i <= n; i++) {
		for (int j = 1; j <= 9; j++) {
			int s = max(j - 2, 1);
			int e = min(j + 2, 9);
			for (int k = s; k <= e; k++) {
				D[i][j] += D[i - 1][k];
				D[i][j] %= 987654321;
			}
		}
	}
	int answer = 0;
	for (int j = 1; j <= 9; j++) {
		answer += D[n][j];
		answer %= 987654321;
	}
	cout << answer;
}