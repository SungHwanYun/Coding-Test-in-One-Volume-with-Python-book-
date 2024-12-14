import java.util.*;

public class Main {
    static int solve(int u, int k, List<Integer> A, List<List<Integer>> E) {
        int ret = A.get(u);
        if (k == 0) return ret;
        for (int v : E.get(u)) {
            ret += solve(v, k - 1, A, E);
        }
        return ret;
    }

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int n = scanner.nextInt();
        int k = scanner.nextInt();
        List<List<Integer>> E = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            E.add(new ArrayList<>());
        }
        for (int i = 1; i < n; i++) {
            int u = scanner.nextInt();
            int v = scanner.nextInt();
            E.get(u).add(v);
        }
        List<Integer> A = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            A.add(scanner.nextInt());
        }
        System.out.println(solve(0, k, A, E));
    }
}
