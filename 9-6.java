import java.util.*;

public class Solution {
    static int[][][] D = new int[52][52][2504];
    static int[][] dd = { {1, 0}, {0, -1}, {0, 1}, {-1, 0} };
    static String dir_str = "dlru";

    static boolean in_range(int r, int c, int n, int m) {
        return 0 <= r && r < n && 0 <= c && c < m;
    }

    static void make_D(int r, int c, int kk, int k, int n, int m) {
        D[r][c][kk] = 1;
        if (kk == k) {
            return;
        }
        for (int[] d : dd) {
            int dr = d[0];
            int dc = d[1];
            int nr = r + dr;
            int nc = c + dc;
            if (in_range(nr, nc, n, m) && D[nr][nc][kk + 1] == 0) {
                make_D(nr, nc, kk + 1, k, n, m);
            }
        }
    }

    static String solution(int n, int m, int x, int y, int r, int c, int k) {
        int dist = Math.abs(r - x) + Math.abs(c - y);
        if (dist > k || ((dist & 0x1) != (k & 0x1))) {
            return "impossible";
        }
        x -= 1;
        y -= 1;
        r -= 1;
        c -= 1;
        make_D(r, c, 0, k, n, m);
        StringBuilder answer = new StringBuilder();
        while (k > 0) {
            int idx = 0;
            for (int[] d : dd) {
                int dx = d[0];
                int dy = d[1];
                int nx = x + dx;
                int ny = y + dy;
                if (in_range(nx, ny, n, m) && D[nx][ny][k - 1] == 1) {
                    answer.append(dir_str.charAt(idx));
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
        return answer.toString();
    }
}
