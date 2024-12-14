import java.util.*;

public class Main {
    static class Pair {
        int first;
        int second;

        Pair(int first, int second) {
            this.first = first;
            this.second = second;
        }
    }

    static int[] dr = {0, 0, -1, -1, -1, 0, 1, 1, 1};
    static int[] dc = {0, -1, -1, 0, 1, 1, 1, 0, -1};
    static int N, M;
    static int[][] A = new int[50][50];
    static int[][] C = new int[50][50];
    static List<Pair> B = new ArrayList<>();

    static boolean inRange(int r, int c) {
        return 0 <= r && r < N && 0 <= c && c < N;
    }

    static void moveCloud(int dd, int ss) {
        List<Pair> V = new ArrayList<>();
        for (Pair p : B) {
            int r = p.first;
            int c = p.second;
            int nr = (r + dr[dd] * ss) % N;
            int nc = (c + dc[dd] * ss) % N;
            if (nr < 0) nr += N;
            if (nc < 0) nc += N;
            V.add(new Pair(nr, nc));
        }
        B.clear();
        B.addAll(V);
    }

    static void addOneWater() {
        for (int[] row : C) {
            Arrays.fill(row, 0);
        }
        for (Pair p : B) {
            int r = p.first;
            int c = p.second;
            C[r][c] = 1;
            A[r][c]++;
        }
        for (Pair p : B) {
            int r = p.first;
            int c = p.second;
            int[] rr = {-1, -1, 1, 1};
            int[] cc = {-1, 1, -1, 1};
            for (int j = 0; j < 4; j++) {
                int nr = r + rr[j];
                int nc = c + cc[j];
                if (inRange(nr, nc) && A[nr][nc] > 0) {
                    A[r][c]++;
                }
            }
        }
        B.clear();
    }

    static void buildCloud() {
        for (int r = 0; r < N; r++) {
            for (int c = 0; c < N; c++) {
                if (A[r][c] >= 2 && C[r][c] == 0) {
                    B.add(new Pair(r, c));
                    A[r][c] -= 2;
                }
            }
        }
    }

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        N = scanner.nextInt();
        M = scanner.nextInt();
        for (int r = 0; r < N; r++) {
            for (int c = 0; c < N; c++) {
                A[r][c] = scanner.nextInt();
            }
        }
        B.add(new Pair(N - 1, 0));
        B.add(new Pair(N - 1, 1));
        B.add(new Pair(N - 2, 0));
        B.add(new Pair(N - 2, 1));
        while (M-- > 0) {
            int d = scanner.nextInt();
            int s = scanner.nextInt();
            moveCloud(d, s);
            addOneWater();
            buildCloud();
        }
        int ans = 0;
        for (int r = 0; r < N; r++) {
            for (int c = 0; c < N; c++) {
                ans += A[r][c];
            }
        }
        System.out.print(ans);
    }
}
