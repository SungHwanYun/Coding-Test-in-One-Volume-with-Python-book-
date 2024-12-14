import java.util.*;

public class Main {
    static long solve(int u, int color, List<List<Integer>> A, List<List<Integer>> E) {
        long ret = A.get(u).get(color);
        for (int v : E.get(u)) {
            ret += solve(v, 1 - color, A, E);
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
        List<List<Integer>> A = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            int w = sc.nextInt();
            int b = sc.nextInt();
            List<Integer> temp = new ArrayList<>();
            temp.add(w);
            temp.add(b);
            A.add(temp);
        }
        System.out.println(Math.min(solve(0, 0, A, E), solve(0, 1, A, E)));
    }
}
