import java.util.ArrayList;
import java.util.List;

public class Solution {
    static List<Integer>[] E;
    static int N, answer;

    public static void solve(int state, int sheep, int wolf, int[] info) {
        if (sheep > answer)
            answer = sheep;
        for (int u = 0; u < N; u++) {
            if ((state & (1 << u)) == 0) continue;
            for (int i = 0; i < E[u].size(); i++) {
                int v = E[u].get(i);
                if ((state & (1 << v)) != 0) continue;
                if (info[v] == 0) {
                    solve(state | (1 << v), sheep + 1, wolf, info);
                }
                else {
                    if (sheep > wolf + 1) {
                        solve(state | (1 << v), sheep, wolf + 1, info);
                    }
                }
            }
        }
    }

    public static int solution(int[] info, int[][] edges) {
        N = info.length;
        E = new ArrayList[N];
        for (int i = 0; i < N; i++) {
            E[i] = new ArrayList<>();
        }
        for (int i = 0; i < edges.length; i++) {
            int p = edges[i][0], c = edges[i][1];
            E[p].add(c);
        }
        solve(1, 1, 0, info);
        return answer;
    }
}