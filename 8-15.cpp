#include <bits/stdc++.h>
using namespace std;

bool in_range(int r, int c) {
    return 0 <= r && r <= 4 && 0 <= c && c <= 4;
}

int get_move_count(vector<vector<int>>& board, int sr, int sc, int tr, int tc) {
    vector<vector<int>> dd = { {0, -1}, {0, 1}, {-1, 0}, {1, 0} };
    vector<vector<int>> visited(5, vector<int>(5, 0));
    vector<vector<int>> dist(5, vector<int>(5, 0));
    queue<vector<int>> q;
    q.push({ sr, sc });
    visited[sr][sc] = 1;
    while (!q.empty()) {
        int r = q.front()[0], c = q.front()[1];
        q.pop();
        if (r == tr && c == tc) {
            return dist[r][c];
        }
        for (auto& d : dd) {
            int nr = r + d[0], nc = c + d[1];
            if (in_range(nr, nc) && visited[nr][nc] == 0 && board[nr][nc] != -1) {
                q.push({ nr, nc });
                dist[nr][nc] = dist[r][c] + 1;
                visited[nr][nc] = 1;
            }
        }
        for (auto& d : dd) {
            int nr = r, nc = c;
            while (true) {
                if (!in_range(nr + d[0], nc + d[1])) {
                    break;
                }
                if (board[nr + d[0]][nc + d[1]] == -1) {
                    break;
                }
                nr += d[0];
                nc += d[1];
                if (board[nr][nc] == 7) {
                    break;
                }
            }
            if (visited[nr][nc] == 0) {
                q.push({ nr, nc });
                dist[nr][nc] = dist[r][c] + 1;
                visited[nr][nc] = 1;
            }
        }
    }
    return -1;
}

int solution(vector<vector<int>>& board, int sr, int sc) {
    vector<vector<int>> source(6);
    for (int r = 0; r < 5; r++) {
        for (int c = 0; c < 5; c++) {
            if (board[r][c] > 0 && board[r][c] < 7) {
                source[board[r][c] - 1] = { r, c };
            }
        }
    }
    sort(source.begin(), source.end());
    int answer = -1;
    do {
        int ret = 0;
        int r = sr, c = sc;
        for (auto& target : source) {
            int nr = target[0], nc = target[1];
            int x = get_move_count(board, r, c, nr, nc);
            if (x == -1) {
                ret = -1;
                break;
            }
            ret += x;
            r = nr;
            c = nc;
        }
        if (ret != -1) {
            if (answer == -1 || answer > ret) {
                answer = ret;
            }
        }
    } while (next_permutation(source.begin(), source.end()));
    return answer;
}

int main() {
    vector<vector<int>> board(5, vector<int>(5));
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            cin >> board[i][j];
        }
    }
    int sr, sc;
    cin >> sr >> sc;
    cout << solution(board, sr, sc) << endl;
}