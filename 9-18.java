import java.util.*;

public class Solution {
    static int[][] E = new int[204][204];
    static int[][] D = new int[204][204];
    static final int INF = (int)1e8;

    public static int solution(int n, int s, int a, int b, int[][] fares) {
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= n; j++) {
                if (i != j) {
                    E[i][j] = INF;
                }
            }
        }
        for (int i = 0; i < fares.length; i++) {
            int u = fares[i][0];
            int v = fares[i][1];
            int w = fares[i][2];
            E[u][v] = E[v][u] = w;
        }
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= n; j++) {
                D[i][j] = E[i][j];
            }
        }
        for (int k = 1; k <= n; k++) {
            for (int i = 1; i <= n; i++) {
                for (int j = 1; j <= n; j++) {
                    if (D[i][k] + D[k][j] < D[i][j]) {
                        D[i][j] = D[i][k] + D[k][j];
                    }
                }
            }
        }
        int answer = D[s][a] + D[s][b];
        for (int k = 1; k <= n; k++) {
            if (s == k) {
                continue;
            }
            int ret = D[s][k] + D[k][a] + D[k][b];
            answer = Math.min(answer, ret);
        }
        return answer;
    }
}