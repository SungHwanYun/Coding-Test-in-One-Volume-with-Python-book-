import java.util.ArrayList;
import java.util.List;

class Solution {
    static List<Integer>[] E;
    static int[][] D;

    public static void solve(int r, int[] sales) {
        int child_sum = 0;
        int diff_mn = (int)2e9;
        int is_zero_larger = 0;
        for (int i = 0; i < E[r].size(); i++) {
            int c = E[r].get(i);
            solve(c, sales);
            child_sum += Math.min(D[c][0], D[c][1]);
            if (D[c][0] >= D[c][1]) {
                is_zero_larger = 1;
            }
            if (D[c][0] <= D[c][1]) {
                diff_mn = Math.min(diff_mn, D[c][1] - D[c][0]);
            }
        }
        D[r][1] = child_sum + sales[r - 1];
        if (E[r].size() == 0) {
            D[r][0] = 0;
        }
        else if (is_zero_larger == 1) {
            D[r][0] = child_sum;
        }
        else {
            D[r][0] = child_sum + diff_mn;
        }
    }

    public static int solution(int[] sales, int[][] links) {
        E = new ArrayList[300004];
        D = new int[300004][2];
        for (int i = 0; i < 300004; i++) {
            E[i] = new ArrayList<>();
        }
        for (int i = 0; i < links.length; i++) {
            int p = links[i][0], c = links[i][1];
            E[p].add(c);
        }
        solve(1, sales);
        return Math.min(D[1][0], D[1][1]);
    }
}
