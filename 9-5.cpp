#include <bits/stdc++.h>
using namespace std;

vector<vector<string>> board(51, vector<string>(51, ""));
vector<vector<vector<int>>> P(51, vector<vector<int>>(51, vector<int>(2, -1)));

vector<int> do_find(vector<int> x) {
    if (P[x[0]][x[1]] == x) {
        return x;
    }
    P[x[0]][x[1]] = do_find(P[x[0]][x[1]]);
    return P[x[0]][x[1]];
}

void do_merge(vector<int> x, vector<int> y) {
    vector<int> px = do_find(x);
    vector<int> py = do_find(y);
    P[py[0]][py[1]] = px;
}

vector<string> solution(vector<string> commands) {
    for (int i = 1; i <= 50; i++) {
        for (int j = 1; j <= 50; j++) {
            P[i][j][0] = i;
            P[i][j][1] = j;
        }
    }
    vector<string> answer;
    for (string c : commands) {
        vector<string> cmd;
        string word = "";
        for (char ch : c) {
            if (ch == ' ') {
                cmd.push_back(word);
                word = "";
            }
            else {
                word += ch;
            }
        }
        cmd.push_back(word);
        if (cmd[0] == "UPDATE" && cmd.size() == 4) {
            vector<int> x = { stoi(cmd[1]), stoi(cmd[2]) };
            vector<int> px = do_find(x);
            board[px[0]][px[1]] = cmd[3];
        }
        else if (cmd[0] == "UPDATE" && cmd.size() == 3) {
            for (int r = 1; r <= 50; r++) {
                for (int c = 1; c <= 50; c++) {
                    if (board[r][c] == cmd[1]) {
                        board[r][c] = cmd[2];
                    }
                }
            }
        }
        else if (cmd[0] == "MERGE") {
            vector<int> x = { stoi(cmd[1]), stoi(cmd[2]) };
            vector<int> y = { stoi(cmd[3]), stoi(cmd[4]) };
            if (x == y) {
                continue;
            }
            vector<int> px = do_find(x);
            vector<int> py = do_find(y);
            string value = "";
            if (board[px[0]][px[1]] == "") {
                value = board[py[0]][py[1]];
            }
            else {
                value = board[px[0]][px[1]];
            }
            board[px[0]][px[1]] = "";
            board[py[0]][py[1]] = "";
            do_merge(px, py);
            board[px[0]][px[1]] = value;
        }
        else if (cmd[0] == "UNMERGE") {
            vector<int> x = { stoi(cmd[1]), stoi(cmd[2]) };
            vector<int> px = do_find(x);
            string ss = board[px[0]][px[1]];
            vector<vector<int>> L;
            for (int r = 1; r <= 50; r++) {
                for (int c = 1; c <= 50; c++) {
                    vector<int> y = do_find({ r, c });
                    if (y == px) {
                        L.push_back({ r, c });
                    }
                }
            }
            for (vector<int> rc : L) {
                P[rc[0]][rc[1]] = { rc[0], rc[1] };
                board[rc[0]][rc[1]] = "";
            }
            board[x[0]][x[1]] = ss;
        }
        else {
            vector<int> x = { stoi(cmd[1]), stoi(cmd[2]) };
            vector<int> px = do_find(x);
            if (board[px[0]][px[1]] == "") {
                answer.push_back("EMPTY");
            }
            else {
                answer.push_back(board[px[0]][px[1]]);
            }
        }
    }
    return answer;
}