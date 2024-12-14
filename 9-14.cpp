#include <bits/stdc++.h>
using namespace std;

int dr[4] = { 0,0,-1,1 }, dc[4] = { -1,1,0,0 };
int N, M;

bool in_range(int r, int c) {
    return 0 <= r && r < N && 0 <= c && c < M;
}

int solve(vector<vector<int>>& board, int r1, int c1, int r2, int c2) {
    vector<int> nxt;

    for (int i = 0; i < 4; i++) {
        int nr = r1 + dr[i], nc = c1 + dc[i];
        if (in_range(nr, nc) == false) continue;
        if (board[nr][nc] == 0) continue;

        board[r1][c1] = 0;
        int ret = solve(board, r2, c2, nr, nc);
        board[r1][c1] = 1;
        nxt.push_back(ret);
    }

    if (nxt.size() == 0) {
        return 0;
    }

    if (r1 == r2 && c1 == c2) {
        return 1;
    }

    sort(nxt.begin(), nxt.end());

    if (nxt[0] > 0) {
        return -(nxt[nxt.size() - 1] + 1);
    }
    else {
        int ret = nxt[0];
        for (int i = 1; i < nxt.size(); i++) {
            if (nxt[i] <= 0)
                ret = max(ret, nxt[i]);
        }
        return -ret + 1;
    }
}

int solution(vector<vector<int>> board, vector<int> aloc, vector<int> bloc) {
    N = board.size(); M = board[0].size();
    int answer = solve(board, aloc[0], aloc[1], bloc[0], bloc[1]);
    return abs(answer);
}