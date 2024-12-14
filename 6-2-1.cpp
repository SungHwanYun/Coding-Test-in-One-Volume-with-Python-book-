#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef pair<int, int> pii;
const ll INF = ll(1e18);
vector<pii> E[100004];
void dijkstra(int start, int end, vector<ll>& dist) {
	dist[start] = 0;
	priority_queue<pair<ll, int>> pq;
	pq.push({ 0, start });
	while (!pq.empty()) {
		ll cost = -pq.top().first;
		int here = pq.top().second;
		pq.pop();

		if (dist[here] < cost) continue;

		for (int i = 0; i < (int)E[here].size(); ++i) {
			int there = E[here][i].first;
			ll next_dist = cost + E[here][i].second;

			if (dist[there] > next_dist) {
				dist[there] = next_dist;
				pq.push(make_pair(-next_dist, there));
			}
		}
	}
}

int main() {
	ios::sync_with_stdio(false);
	cin.tie(NULL);
	cout.tie(NULL);

	int N, M, X, Z;
	int u, v, w;
	cin >> N >> M;
	while (M-- > 0) {
		cin >> u >> v >> w;
		E[u].push_back({ v,w });
	}
	cin >> X >> Z;
	vector<ll> dist(N + 1, INF);
	dijkstra(X, Z, dist);
	if (dist[Z] >= INF) dist[Z] = -1;
	cout << dist[Z];
}