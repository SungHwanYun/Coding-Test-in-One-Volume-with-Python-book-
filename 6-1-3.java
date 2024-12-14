import java.util.*;

class Main {
    static class Pair {
        int first;
        int second;

        Pair(int first, int second) {
            this.first = first;
            this.second = second;
        }
    }

    static boolean inRange(int r, int c) {
        return 0 <= r && r <= 4 && 0 <= c && c <= 4;
    }

    static int getMoveCount(int[][] A, int sr, int sc, int tr, int tc) {
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
            int r = now.first;
            int c = now.second;
            if (r == tr && c == tc) {
                return dist[r][c];
            }

            for (int i = 0; i < 4; i++) {
                int nr = r + dr[i];
                int nc = c + dc[i];
                if (inRange(nr, nc) && visited[nr][nc] == 0 && A[nr][nc] != -1) {
                    Q.add(new Pair(nr, nc));
                    dist[nr][nc] = dist[r][c] + 1;
                    visited[nr][nc] = 1;
                }
            }

            for (int i = 0; i < 4; i++) {
                int nr = r;
                int nc = c;
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
        int tr = 0;
        int tc = 0;
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                if (A[i][j] == 1) {
                    tr = i;
                    tc = j;
                }
            }
        }
        System.out.println(getMoveCount(A, sr, sc, tr, tc));
    }
}
