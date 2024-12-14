import java.util.*;

public class Main {
    public static void main(String[] args) {
        Scanner scan = new Scanner(System.in);
        int[][] A = new int[5][5];
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                A[i][j] = scan.nextInt();
            }
        }
        int sr = scan.nextInt();
        int sc = scan.nextInt();
        int tr = 0, tc = 0;
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                if (A[i][j] == 1) {
                    tr = i;
                    tc = j;
                }
            }
        }
        int[] dr = { -1, 1, 0, 0 };
        int[] dc = { 0, 0, -1, 1 };
        int[][] visited = new int[5][5];
        int[][] dist = new int[5][5];
        Deque<Pair<Integer, Integer>> Q = new ArrayDeque<>();
        Q.add(new Pair<>(sr, sc));
        visited[sr][sc] = 1;
        dist[sr][sc] = 0;
        while (!Q.isEmpty()) {
            Pair<Integer, Integer> now = Q.pollFirst();
            int r = now.getFirst();
            int c = now.getSecond();
            if (r == tr && c == tc) {
                System.out.println(dist[r][c]);
                System.exit(0);
            }
            for (int i = 0; i < 4; i++) {
                int nr = r + dr[i];
                int nc = c + dc[i];
                if (inRange(nr, nc) && visited[nr][nc] == 0 && A[nr][nc] != -1) {
                    Q.add(new Pair<>(nr, nc));
                    dist[nr][nc] = dist[r][c] + 1;
                    visited[nr][nc] = 1;
                }
            }
        }
        System.out.println(-1);
    }

    public static boolean inRange(int r, int c) {
        return 0 <= r && r <= 4 && 0 <= c && c <= 4;
    }
}

class Pair<T, U> {
    private T first;
    private U second;

    public Pair(T first, U second) {
        this.first = first;
        this.second = second;
    }

    public T getFirst() {
        return first;
    }

    public U getSecond() {
        return second;
    }
}
