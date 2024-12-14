#include <bits/stdc++.h>
using namespace std;

void update_board(vector<vector<int>>& board, int r1, int c1, int r2, int c2, int degree) {
    for (int r = r1; r <= r2; r++) {
        board[r][c1] += degree;
        if (c2 + 1 < board[0].size())
            board[r][c2 + 1] -= degree;
    }
}
int solution(vector<vector<int>> board, vector<vector<int>> skill) {
    int answer = 0;

    // skill ����� board_diff�� �ݿ��Ѵ�.
    vector<vector<int>> board_diff(board.size());

    // board_diff�� 0���� �ʱ�ȭ�Ѵ�.
    for (int i = 0; i < board.size(); i++) {
        board_diff[i] = vector<int>(board[0].size(), 0);
    }

    // skill����� board_diff�� �ݿ��Ѵ�.
    for (int i = 0; i < skill.size(); i++) {
        update_board(board_diff, skill[i][1], skill[i][2], skill[i][3], skill[i][4], skill[i][0] == 1 ? -skill[i][5] : skill[i][5]);
    }
    for (int r = 0; r < board_diff.size(); r++) {
        for (int c = 1; c < board_diff[0].size(); c++) {
            board_diff[r][c] += board_diff[r][c - 1];
        }
        for (int c = 0; c < board_diff[0].size(); c++)
            board[r][c] += board_diff[r][c];
    }

    for (int i = 0; i < board.size(); i++) {
        for (int j = 0; j < board[i].size(); j++) {
            if (board[i][j] > 0) answer++;
        }
    }
    return answer;
}