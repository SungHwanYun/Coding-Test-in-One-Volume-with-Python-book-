import java.util.*;

public class Main {
    static long solve(int u, List<Integer> A, List<List<Integer>> E) {
        long ret = A.get(u);
        for (int v : E.get(u)) {
            long ret2 = solve(v, A, E);
            if (ret2 > 0) ret += ret2;
        }
        return ret;
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();
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
        System.out.println(solve(0, A, E));
    }
}
