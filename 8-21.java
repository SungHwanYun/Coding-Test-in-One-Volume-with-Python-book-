import java.util.*;

class Main {
    public static int solve(int state, int k, int apple, List<Integer> A, List<List<Integer>> E, List<Integer> visited) {
        if (visited.get(state) == 1) {
            return 0;
        }
        visited.set(state, 1);
        int ret = apple;
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
                ret = Math.max(ret, solve(state | (1 << v), k - 1, apple + A.get(v), A, E, visited));
            }
        }
        return ret;
    }

    public static int solution(int n, int k, List<Integer> A, List<List<Integer>> edges) {
        List<List<Integer>> E = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            E.add(new ArrayList<>());
        }
        for (List<Integer> edge : edges) {
            int p = edge.get(0);
            int c = edge.get(1);
            E.get(p).add(c);
        }
        List<Integer> visited = new ArrayList<>(Collections.nCopies(1 << n, 0));
        return solve(1 << 0, k - 1, A.get(0), A, E, visited);
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
        System.out.println(solution(n, k, A, edges));
    }
}
