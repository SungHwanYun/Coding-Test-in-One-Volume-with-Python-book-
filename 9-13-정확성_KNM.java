import java.util.*;

class Solution {
    public static void update_board(int[][] board, int r1, int c1, int r2, int c2, int degree) {
        for (int r = r1; r <= r2; r++) 
            for (int c = c1; c <= c2; c++)
                board[r][c] += degree;
    }

    public static int solution(int[][] board, int[][] skill) {
        int answer = 0;
        for (int i = 0; i < skill.length; i++) {
            update_board(board, skill[i][1], skill[i][2], skill[i][3], skill[i][4], skill[i][0] == 1 ? -skill[i][5] : skill[i][5]);
        }
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[i].length; j++) {
                if (board[i][j] > 0) answer++;
            }
        }
        return answer;
    }
}