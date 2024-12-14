import java.util.*;

public class Main {
    public static List<Integer> solve(int state, int k, int apple, int pear, List<Integer> A, List<List<Integer>> E, List<Integer> visited) {
        if (visited.get(state) == 1) {
            return Arrays.asList(0, 0);
        }
        visited.set(state, 1);
        List<Integer> ret = Arrays.asList(apple, pear);
        if (k == 0) {
            return ret;
        }
        for (int u = 0; u < A.size(); u++) {
            if ((state & (1 << u)) == 0) {
                continue;
            }
            for (int v : E.get(u)) {
                if ((state & (1 << v)) != 0) {
                    continue;
                }
                List<Integer> ret2 = solve(state | (1 << v), k - 1, apple + (A.get(v) == 1 ? 1 : 0), pear + (A.get(v) == 2 ? 1 : 0), A, E, visited);
                if (ret2.get(0) * ret2.get(1) > ret.get(0) * ret.get(1) ||
                    (ret2.get(0) * ret2.get(1) == ret.get(0) * ret.get(1) && ret2.get(0) > ret.get(0)) ||
                    (ret2.get(0) * ret2.get(1) == ret.get(0) * ret.get(1) && ret2.get(0) == ret.get(0) && ret2.get(1) > ret.get(1))) {
                    ret.set(0, ret2.get(0));
                    ret.set(1, ret2.get(1));
                }
            }
        }
        return ret;
    }

    public static List<Integer> solution(int n, int k, List<Integer> A, List<List<Integer>> edges) {
        List<List<Integer>> E = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            E.add(new ArrayList<>());
        }
        for (List<Integer> edge : edges) {
            E.get(edge.get(0)).add(edge.get(1));
        }
        List<Integer> visited = new ArrayList<>();
        for (int i = 0; i < (1 << n); i++) {
            visited.add(0);
        }
        return solve(1 << 0, k - 1, (A.get(0) == 1 ? 1 : 0), (A.get(0) == 2 ? 1 : 0), A, E, visited);
    }

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int n = scanner.nextInt();
        int k = scanner.nextInt();
        List<List<Integer>> edges = new ArrayList<>();
        for (int i = 0; i < n - 1; i++) {
            List<Integer> edge = new ArrayList<>();
            edge.add(scanner.nextInt());
            edge.add(scanner.nextInt());
            edges.add(edge);
        }
        List<Integer> A = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            A.add(scanner.nextInt());
        }
        List<Integer> ret = solution(n, k, A, edges);
        System.out.println(ret.get(0) + " " + ret.get(1));
    }
}
