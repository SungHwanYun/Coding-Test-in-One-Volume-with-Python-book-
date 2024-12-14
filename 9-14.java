import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

class Solution {
    private static int[] dr = {0, 0, -1, 1};
    private static int[] dc = {-1, 1, 0, 0};
    private static int N, M;

    private static boolean inRange(int r, int c) {
        return 0 <= r && r < N && 0 <= c && c < M;
    }

    private static int solve(int[][] board, int r1, int c1, int r2, int c2) {
        List<Integer> nxt = new ArrayList<>();
        for (int i = 0; i < 4; i++) {
            int nr = r1 + dr[i], nc = c1 + dc[i];
            if (!inRange(nr, nc)) continue;
            if (board[nr][nc] == 0) continue;
            board[r1][c1] = 0;
            int ret = solve(board, r2, c2, nr, nc);
            board[r1][c1] = 1;
            nxt.add(ret);
        }
        if (nxt.size() == 0) {
            return 0;
        }
        if (r1 == r2 && c1 == c2) {
            return 1;
        }
        Collections.sort(nxt);
        if (nxt.get(0) > 0) {
            return -(nxt.get(nxt.size() - 1) + 1);
        } else {
            int ret = nxt.get(0);
            for (int i = 1; i < nxt.size(); i++) {
                if (nxt.get(i) <= 0) {
                    ret = Math.max(ret, nxt.get(i));
                }
            }
            return -ret + 1;
        }
    }

    public static int solution(int[][] board, int[] aloc, int[] bloc) {
        N = board.length;
        M = board[0].length;
        int answer = solve(board, aloc[0], aloc[1], bloc[0], bloc[1]);
        return Math.abs(answer);
    }
}
