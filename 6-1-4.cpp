#include<bits/stdc++.h>
using namespace std;
typedef pair<int, int> pii;
int in_range(int r, int c) {
    return 0 <= r && r <= 4 && 0 <= c && c <= 4;
}
int get_move_count(int A[5][5], int sr, int sc, int tr, int tc) {
    int dr[4] = { -1, 1, 0, 0 }, dc[4] = { 0, 0, -1, 1 };
    int visited[5][5] = { 0 }, dist[5][5] = { 0 };
    deque<pii> Q;
    Q.push_back({ sr, sc });
    visited[sr][sc] = 1;
    dist[sr][sc] = 0;
    while (!Q.empty()) {
        pii now = Q.front(); Q.pop_front();
        int r = now.first, c = now.second;
        if (r == tr && c == tc) {
            return dist[r][c];
        }

        // 걸어간다.
        for (int i = 0; i < 4; i++) {
            int nr = r + dr[i], nc = c + dc[i];
            if (in_range(nr, nc) && visited[nr][nc] == 0 && A[nr][nc] != -1) {
                Q.push_back({ nr,nc });
                dist[nr][nc] = dist[r][c] + 1;
                visited[nr][nc] = 1;
            }
        }

        // 뛰어간다.
        for (int i = 0; i < 4; i++) {
            int nr = r, nc = c;
            while (1) {
                if (!in_range(nr + dr[i], nc + dc[i])) break;
                if (A[nr + dr[i]][nc + dc[i]] == -1) break;
                nr += dr[i]; nc += dc[i];
                if (A[nr][nc] == 7) break;
            }
            if (visited[nr][nc] == 0) {
                Q.push_back({ nr,nc });
                dist[nr][nc] = dist[r][c] + 1;
                visited[nr][nc] = 1;
            }
        }
    }
    return -1;
}
int main(int argc, char* argv[]) {
    ios::sync_with_stdio(false);
    cin.tie(NULL);
    cout.tie(NULL);

    int A[5][5];
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            cin >> A[i][j];
        }
    }

    int sr, sc; cin >> sr >> sc;
    pii target[6];
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            if (A[i][j] > 0 && A[i][j] < 7) {
                target[A[i][j] - 1] = { i, j };
            }
        }
    }

    int answer = 0;
    int r = sr, c = sc;
    for (int i = 0; i < 6; i++) {
        int nr = target[i].first, nc = target[i].second;
        int ret = get_move_count(A, r, c, nr, nc);
        if (ret == -1) {
            cout << -1; exit(0);
        }
        answer += ret;
        r = nr; c = nc;
    }
    cout << answer;
}