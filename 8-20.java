import java.util.*;

public class Main {
    static class Pair {
        int first;
        int second;
        
        public Pair(int first, int second) {
            this.first = first;
            this.second = second;
        }
    }
    
    static int n;
    static List<Pair> edges;
    static int[][] A;
    static List<Integer>[] E;
    static long[][] D;
    
    static long solve(int u, int color) {
        if (D[u][color] != -1) return D[u][color];
        D[u][color] = A[u][color];
        for (int v : E[u]) {
            if (color == 0) {
                D[u][color] += Math.min(solve(v, 0), solve(v, 1));
            }
            else {
                D[u][color] += solve(v, 0);
            }
        }
        return D[u][color];
    }
    
    static long solution() {
        for (Pair e : edges) {
            int p = e.first, c = e.second;
            E[p].add(c);
        }
        for (int i = 0; i < n; i++) D[i][0] = D[i][1] = -1;
        return Math.min(solve(0, 0), solve(0, 1));
    }
    
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        n = sc.nextInt();
        edges = new ArrayList<>();
        A = new int[n][2];
        E = new ArrayList[n];
        for (int i = 0; i < n; i++) {
            E[i] = new ArrayList<>();
        }
        D = new long[n][2];
        for (int i = 0; i < n; i++) {
            D[i][0] = D[i][1] = -1;
        }
        for (int i = 1; i < n; i++) {
            int a = sc.nextInt();
            int b = sc.nextInt();
            edges.add(new Pair(a, b));
        }
        for (int i = 0; i < n; i++) {
            A[i][0] = sc.nextInt();
            A[i][1] = sc.nextInt();
        }
        System.out.println(solution());
    }
}
