#include <bits/stdc++.h>
using namespace std;

int solve(vector<vector<int>>& board, vector<int> aloc, vector<int> bloc, int apple_diff);
int in_range(vector<int> loc);

int solution(vector<vector<int>>& board, vector<int> aloc, vector<int> bloc) {
    return solve(board, aloc, bloc, 0);
}

int solve(vector<vector<int>>& board, vector<int> aloc, vector<int> bloc, int apple_diff) {
    if (board[aloc[0]][aloc[1]] == -1 && board[bloc[0]][bloc[1]] == -1) {
        if (apple_diff > 0) {
            return 1;
        }
        return 0;
    }
    int remained_apple = 0;
    for (int i = 0; i < 5; i++) {
        remained_apple += count(board[i].begin(), board[i].end(), 1);
    }
    if (remained_apple == 0) {
        if (apple_diff > 0) {
            return 1;
        }
        return 0;
    }
    vector<vector<int>> dd = { {-1, 0}, {1, 0}, {0, -1}, {0, 1} };
    int try_count = 0;
    for (auto& d : dd) {
        int dr = d[0];
        int dc = d[1];
        int r = aloc[0] + dr;
        int c = aloc[1] + dc;
        if (in_range(vector<int>({ r, c })) && board[r][c] != -1 && vector<int>({ r, c }) != bloc) {
            try_count += 1;
            int prv_value = board[aloc[0]][aloc[1]];
            board[aloc[0]][aloc[1]] = -1;
            int ret = solve(board, bloc, vector<int>({ r, c }), -(apple_diff + board[r][c]) + 1);
            board[aloc[0]][aloc[1]] = prv_value;
            if (ret == 0) {
                return 1;
            }
        }
    }
    if (try_count == 0) {
        int prv_value = board[aloc[0]][aloc[1]];
        board[aloc[0]][aloc[1]] = -1;
        int ret = solve(board, bloc, aloc, -apple_diff + 1);
        board[aloc[0]][aloc[1]] = prv_value;
        if (ret == 0) {
            return 1;
        }
    }
    return 0;
}

int in_range(vector<int> loc) {
    return 0 <= loc[0] && loc[0] <= 4 && 0 <= loc[1] && loc[1] <= 4;
}

int main() {
    vector<vector<int>> board;
    for (int i = 0; i < 5; i++) {
        vector<int> row;
        for (int j = 0; j < 5; j++) {
            int num;
            cin >> num;
            row.push_back(num);
        }
        board.push_back(row);
    }
    vector<int> loc;
    for (int i = 0; i < 4; i++) {
        int num;
        cin >> num;
        loc.push_back(num);
    }
    vector<int> aloc(loc.begin(), loc.begin() + 2);
    vector<int> bloc(loc.begin() + 2, loc.end());
    cout << solution(board, aloc, bloc) << endl;
    return 0;
}