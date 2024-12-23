import java.util.*;

class Solution {
    public static void update_board(int[][] board, int r1, int c1, int r2, int c2, int degree) {
        board[r1][c1] += degree;
        if (c2 + 1 < board[0].length)
            board[r1][c2 + 1] -= degree;
        if (r2 + 1 < board.length)
            board[r2 + 1][c1] -= degree;
        if (r2 + 1 < board.length && c2 + 1 < board[0].length)
            board[r2 + 1][c2 + 1] += degree;
    }

    public static int solution(int[][] board, int[][] skill) {
        int answer = 0;
        int[][] board_diff = new int[board.length][board[0].length];
        for (int i = 0; i < board.length; i++)
            for (int j = 0; j < board[i].length; j++)
                board_diff[i][j] = 0;
        for (int i = 0; i < skill.length; i++) {
            update_board(board_diff, skill[i][1], skill[i][2], skill[i][3], skill[i][4], skill[i][0] == 1 ? -skill[i][5] : skill[i][5]);
        }
        for (int r = 0; r < board_diff.length; r++) {
            for (int c = 1; c < board_diff[0].length; c++) {
                board_diff[r][c] += board_diff[r][c - 1];
            }
        }
        for (int c = 0; c < board_diff[0].length; c++) {
            for (int r = 1; r < board_diff.length; r++) {
                board_diff[r][c] += board_diff[r - 1][c];
            }
        }
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[i].length; j++) {
                if (board[i][j] + board_diff[i][j] > 0) answer++;
            }
        }
        return answer;
    }
}