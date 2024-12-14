import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

class Solution {
    static int n = 0;
    static List<List<Integer>> E = new ArrayList<>(102);
    static List<Integer> L = new ArrayList<>();
    static List<Integer> X = new ArrayList<>(Collections.nCopies(102, -1));
    static List<Integer> Y = new ArrayList<>();

    static List<Integer> assign_stone(int k, List<Integer> Y, int[] target) {
        List<Integer> cnt = new ArrayList<>(Collections.nCopies(n, 0));
        for (int i = 0; i < k; i++) {
            cnt.set(Y.get(i), cnt.get(Y.get(i)) + 1);
        }
        List<Integer> answer = new ArrayList<>();
        for (int i = 0; i < k; i++) {
            int u = Y.get(i);
            cnt.set(u, cnt.get(u) - 1);
            if (target[u] - 1 <= cnt.get(u) * 3) {
                answer.add(1);
                target[u] -= 1;
            } else if (target[u] - 2 <= cnt.get(u) * 3) {
                answer.add(2);
                target[u] -= 2;
            } else {
                answer.add(3);
                target[u] -= 3;
            }
        }
        return answer;
    }

    static boolean is_ok(int k, List<Integer> Y, int[] target) {
        List<Integer> cnt = new ArrayList<>(Collections.nCopies(n, 0));
        for (int i = 0; i < k; i++) {
            cnt.set(Y.get(i), cnt.get(Y.get(i)) + 1);
        }
        for (int i = 0; i < n; i++) {
            if (target[i] < cnt.get(i) || cnt.get(i) * 3 < target[i]) {
                return false;
            }
        }
        return true;
    }

    static int dfs(int u) {
        if (E.get(u).isEmpty()) {
            return u;
        }
        int ret = dfs(E.get(u).get(X.get(u)));
        X.set(u, (X.get(u) + 1) % E.get(u).size());
        return ret;
    }

    static int[] solution(int[][] edges, int[] target) {
        n = edges.length + 1;
        for (int i = 0; i < 102; i++) {
            E.add(new ArrayList<>());
        }
        for (int[] edge : edges) {
            E.get(edge[0] - 1).add(edge[1] - 1);
        }
        for (int i = 0; i < n; i++) {
            Collections.sort(E.get(i));
            if (E.get(i).isEmpty()) {
                L.add(i);
            } else {
                X.set(i, 0);
            }
        }
        for (int i = 0; i < 10004; i++) {
            Y.add(dfs(0));
        }
        List<Integer> answer = new ArrayList<>();
        answer.add(-1);
        for (int k = 1; k < 10004; k++) {
            if (is_ok(k, Y, target)) {
                answer = assign_stone(k, Y, target);
                break;
            }
        }
        int[] ans = new int[answer.size()];
        for (int i = 0; i < ans.length; i++) ans[i] = answer.get(i);
        return ans;
    }
}