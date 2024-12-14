#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef pair<int, int> pii;
const ll INF = ll(1e18);
int N, X, Y, Z;
vector<pii> E[100004];
bool selected[100004];
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

int main(void) {
	int q, u, v, x, i;
	scanf("%d%d\n", &N, &q);
	while (q-- > 0) {
		scanf("%d%d%d\n", &u, &v, &x);
		E[u].push_back({ v,x });
		E[v].push_back({ u,x });
	}
	scanf("%d%d\n", &X, &Z);
	scanf("%d\n", &q);
	vector<int> P(q);
	for (auto& p : P)
		scanf("%d", &p);

	vector<ll> dist_X(N + 1, INF);
	for (i = 1; i <= N; i++) selected[i] = false;
	dijkstra(X, Z, dist_X);

	vector<ll> dist_Z(N + 1, INF);
	for (i = 1; i <= N; i++) selected[i] = false;
	dijkstra(Z, X, dist_Z);

	ll ans = INF;
	for (auto p : P) {
		if (dist_X[p] < INF && dist_Z[p]<INF && ans>dist_X[p] + dist_Z[p])
			ans = dist_X[p] + dist_Z[p];
	}

	if (ans >= INF) ans = -1;
	printf("%lld\n", ans);
}