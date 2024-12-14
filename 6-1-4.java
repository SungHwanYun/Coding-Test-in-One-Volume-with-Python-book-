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

    public static boolean inRange(int r, int c) {
        return 0 <= r && r <= 4 && 0 <= c && c <= 4;
    }

    public static int getMoveCount(int[][] A, int sr, int sc, int tr, int tc) {
        int[] dr = { -1, 1, 0, 0 };
        int[] dc = { 0, 0, -1, 1 };
        int[][] visited = new int[5][5];
        int[][] dist = new int[5][5];
        Deque<Pair> Q = new ArrayDeque<>();
        Q.add(new Pair(sr, sc));
        visited[sr][sc] = 1;
        dist[sr][sc] = 0;
        while (!Q.isEmpty()) {
            Pair now = Q.pollFirst();
            int r = now.first, c = now.second;
            if (r == tr && c == tc) {
                return dist[r][c];
            }
            for (int i = 0; i < 4; i++) {
                int nr = r + dr[i], nc = c + dc[i];
                if (inRange(nr, nc) && visited[nr][nc] == 0 && A[nr][nc] != -1) {
                    Q.add(new Pair(nr, nc));
                    dist[nr][nc] = dist[r][c] + 1;
                    visited[nr][nc] = 1;
                }
            }
            for (int i = 0; i < 4; i++) {
                int nr = r, nc = c;
                while (true) {
                    if (!inRange(nr + dr[i], nc + dc[i]))
                        break;
                    if (A[nr + dr[i]][nc + dc[i]] == -1)
                        break;
                    nr += dr[i];
                    nc += dc[i];
                    if (A[nr][nc] == 7)
                        break;
                }
                if (visited[nr][nc] == 0) {
                    Q.add(new Pair(nr, nc));
                    dist[nr][nc] = dist[r][c] + 1;
                    visited[nr][nc] = 1;
                }
            }
        }
        return -1;
    }

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int[][] A = new int[5][5];
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                A[i][j] = scanner.nextInt();
            }
        }
        int sr = scanner.nextInt();
        int sc = scanner.nextInt();
        Pair[] target = new Pair[6];
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                if (A[i][j] > 0 && A[i][j] < 7) {
                    target[A[i][j] - 1] = new Pair(i, j);
                }
            }
        }
        int answer = 0;
        int r = sr, c = sc;
        for (int i = 0; i < 6; i++) {
            int nr = target[i].first, nc = target[i].second;
            int ret = getMoveCount(A, r, c, nr, nc);
            if (ret == -1) {
                System.out.println(-1);
                System.exit(0);
            }
            answer += ret;
            r = nr;
            c = nc;
        }
        System.out.println(answer);
    }
}