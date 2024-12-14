#include <bits/stdc++.h>
using namespace std;
typedef pair<int, int> pii;

int dr[4] = { 0,0,-1,1 }, dc[4] = { -1,1,0,0 };
pii A[10], B[10];
vector<int> X, Y;
int N, sr, sc, S[10];
int answer = (int)1e8;
  
bool in_range(int r, int c) {
    return 0 <= r && r < 4 && 0 <= c && c < 4;
}

int bfs(vector<vector<int>>& board, int r1, int c1, int r2, int c2) {
    int visited[4][4] = { 0, }, dist[4][4];
    queue<pii> Q;
    Q.push({ r1,c1 });
    dist[r1][c1] = 0;
    visited[r1][c1] = 1;

    while (!Q.empty()) {
        int r = Q.front().first, c = Q.front().second, nr, nc;
        Q.pop();

        if (r == r2 && c == c2) {
            return dist[r][c] + 1;
        }

        for (int i = 0; i < 4; i++) {
            nr = r + dr[i]; nc = c + dc[i];
            if (in_range(nr, nc) && !visited[nr][nc]) {
                Q.push({ nr,nc });
                dist[nr][nc] = dist[r][c] + 1;
                visited[nr][nc] = 1;
            }
        }

        for (int i = 0; i < 4; i++) {
            nr = r; nc = c;
            do {
                if (in_range(nr + dr[i], nc + dc[i]) == false)
                    break;
                nr += dr[i]; nc += dc[i];
                if (board[nr][nc] != 0) break;
            } while (1);

            if (!visited[nr][nc]) {
                Q.push({ nr,nc });
                dist[nr][nc] = dist[r][c] + 1;
                visited[nr][nc] = 1;
            }
        }
    }
    return (int)1e8;
}

int get_move_count(vector<vector<int>> board) {
    int sum = 0, r = sr, c = sc;
    for (int i = 0; i < Y.size(); ++i) {
        int x = Y[i], nr1, nc1, nr2, nc2;
        if (x < 10) {
            x = X[x];
            nr1 = A[x].first; nc1 = A[x].second;
            nr2 = B[x].first; nc2 = B[x].second;
        }
        else {
            x = X[x - 10];
            nr1 = B[x].first; nc1 = B[x].second;
            nr2 = A[x].first; nc2 = A[x].second;
        }
        sum += bfs(board, r, c, nr1, nc1) + bfs(board, nr1, nc1, nr2, nc2);
        r = nr2; c = nc2;
        board[nr1][nc1] = board[nr2][nc2] = 0;
    }
    return sum;
}

void solve(vector<vector<int>>& board) {
    if (Y.size() == N) {
        answer = min(answer, get_move_count(board));
        return;
    }

    for (int i = 0; i < N; i++) {
        if (S[i] == 1) continue;
        S[i] = 1;
        Y.push_back(i); solve(board); Y.pop_back();
        Y.push_back(10 + i); solve(board); Y.pop_back();
        S[i] = 0;
    }
}

int solution(vector<vector<int>> board, int r, int c) {
    sr = r; sc = c;

    int i, j, cnt[10] = { 0, };
    for (i = 0; i < 4; i++) {
        for (j = 0; j < 4; j++) {
            int c = board[i][j];
            if (c != 0) {
                cnt[c]++;
                if (cnt[c] == 1) {
                    A[c] = { i, j };
                }
                else {
                    B[c] = { i, j };
                    X.push_back(c);
                }
            }
        }
    }
    sort(X.begin(), X.end());
    N = X.size();

    solve(board);
    return answer;
}