import java.util.*;

public class Main {
    static int solve(int u, int depth, int k, List<Integer> A, List<List<Integer>> E) {
        if (A.get(u) == k) return depth;
        for (int v : E.get(u)) {
            int ret = solve(v, depth + 1, k, A, E);
            if (ret != -1) return ret;
        }
        return -1;
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();
        int k = sc.nextInt();
        List<List<Integer>> E = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            E.add(new ArrayList<>());
        }
        for (int i = 1; i < n; i++) {
            int u = sc.nextInt();
            int v = sc.nextInt();
            E.get(u).add(v);
        }
        List<Integer> A = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            A.add(sc.nextInt());
        }
        System.out.println(solve(0, 0, k, A, E));
    }
}
