#include <bits/stdc++.h>
using namespace std;

vector<vector<vector<int>>> D(52, vector<vector<int>>(52, vector<int>(2504, 0)));
vector<vector<int>> dd = { {1, 0}, {0, -1}, {0, 1}, {-1, 0} };
string dir_str = "dlru";

bool in_range(int r, int c, int n, int m) {
    return 0 <= r && r < n && 0 <= c && c < m;
}

void make_D(int r, int c, int kk, int k, int n, int m) {
    D[r][c][kk] = 1;
    if (kk == k) {
        return;
    }
    for (auto& d : dd) {
        int dr = d[0];
        int dc = d[1];
        int nr = r + dr;
        int nc = c + dc;
        if (in_range(nr, nc, n, m) && D[nr][nc][kk + 1] == 0) {
            make_D(nr, nc, kk + 1, k, n, m);
        }
    }
}

string solution(int n, int m, int x, int y, int r, int c, int k) {
    int dist = abs(r - x) + (c - y);
    if (dist > k || ((dist & 0x1) != (k & 0x1))) {
        return "impossible";
    }
    x -= 1;
    y -= 1;
    r -= 1;
    c -= 1;
    make_D(r, c, 0, k, n, m);
    string answer = "";
    while (k > 0) {
        int idx = 0;
        for (auto& d : dd) {
            int dx = d[0];
            int dy = d[1];
            int nx = x + dx;
            int ny = y + dy;
            if (in_range(nx, ny, n, m) && D[nx][ny][k - 1] == 1) {
                answer += dir_str[idx];
                x = nx;
                y = ny;
                break;
            }
            idx += 1;
        }
        if (idx == 4) {
            return "impossible";
        }
        k -= 1;
    }
    return answer;
}