import java.util.Scanner;

public class Main {
    static int[] dr = {0, 1, 0, -1};
    static int[] dc = {1, 0, -1, 0};
    static int[][] A, B, C, visited;
    static int N, M, K;

    static boolean inRange(int r, int c) {
        return 1 <= r && r <= N && 1 <= c && c <= M;
    }

    static int dfs(int r, int c) {
        int ret = 1;
        visited[r][c] = 1;
        for (int i = 0; i < 4; i++) {
            int nr = r + dr[i], nc = c + dc[i];
            if (inRange(nr, nc) && visited[nr][nc] == 0 && A[r][c] == A[nr][nc])
                ret += dfs(nr, nc);
        }
        return ret;
    }

    static int getNextPos(int r, int c, int[] nextPos, int d) {
        nextPos[0] = r + dr[d];
        nextPos[1] = c + dc[d];
        if (!inRange(nextPos[0], nextPos[1])) {
            d = (d + 2) % 4;
            nextPos[0] = r + dr[d];
            nextPos[1] = c + dc[d];
        }
        return d;
    }

    static void rotateCube(int d) {
        int t;
        if (d == 0) {
            t = C[4][2];
            C[4][2] = C[2][3];
            C[2][3] = C[2][2];
            C[2][2] = C[2][1];
            C[2][1] = t;
        } else if (d == 2) {
            t = C[4][2];
            C[4][2] = C[2][1];
            C[2][1] = C[2][2];
            C[2][2] = C[2][3];
            C[2][3] = t;
        } else if (d == 1) {
            t = C[4][2];
            C[4][2] = C[3][2];
            C[3][2] = C[2][2];
            C[2][2] = C[1][2];
            C[1][2] = t;
        } else {
            t = C[4][2];
            C[4][2] = C[1][2];
            C[1][2] = C[2][2];
            C[2][2] = C[3][2];
            C[3][2] = t;
        }
    }

    static int updateDir(int r, int c, int[] d) {
        int a = C[4][2], b = A[r][c];
        if (a > b) {
            d[0] = (d[0] + 1) % 4;
        } else if (a < b) {
            d[0] = (d[0] - 1 + 4) % 4;
        }
        return d[0];
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        N = sc.nextInt();
        M = sc.nextInt();
        K = sc.nextInt();
        A = new int[N + 1][M + 1];
        B = new int[N + 1][M + 1];
        C = new int[6][6];
        visited = new int[N + 1][M + 1];

        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= M; j++) {
                A[i][j] = sc.nextInt();
            }
        }

        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= M; j++) {
                for (int r = 1; r <= N; r++)
                    for (int c = 1; c <= M; c++)
                        visited[r][c] = 0;
                B[i][j] = dfs(i, j) * A[i][j];
            }
        }

        C[1][2] = 2;
        C[2][1] = 4;
        C[2][2] = 1;
        C[2][3] = 3;
        C[3][2] = 5;
        C[4][2] = 6;

        int ans = 0, r = 1, c = 1, d = 0;
        for (int k = 0; k < K; k++) {
            int[] nextPos = new int[2];
            d = getNextPos(r, c, nextPos, d);
            ans += B[nextPos[0]][nextPos[1]];
            rotateCube(d);
            r = nextPos[0];
            c = nextPos[1];
            d = updateDir(r, c, new int[]{d});
        }
        System.out.println(ans);
    }
}
