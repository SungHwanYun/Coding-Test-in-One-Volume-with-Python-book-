import java.util.*;

enum DIR {
    RIGHT(1), LEFT(2), UP(3), DOWN(4);
    private final int value;
    DIR(int value) {
        this.value = value;
    }
    public int getValue() {
        return value;
    }
}

enum DIR_MASK {
    R_MASK(1 << DIR.RIGHT.getValue()), L_MASK(1 << DIR.LEFT.getValue()), U_MASK(1 << DIR.UP.getValue()), D_MASK(1 << DIR.DOWN.getValue());
    private final int value;
    DIR_MASK(int value) {
        this.value = value;
    }
    public int getValue() {
        return value;
    }
}

class qinfo {
    int r, c, t;
    qinfo(int r1, int c1, int t1) {
        r = r1;
        c = c1;
        t = t1;
    }
}

class Main {
    static int[][] A, B, C, W;
    static int N, M, K;
    static int[] dr = {0, 0, 0, -1, 1};
    static int[] dc = {0, 1, -1, 0, 0};

    static boolean in_range(int r, int c) {
        return 1 <= r && r <= N && 1 <= c && c <= M;
    }

    static void build_B_sub(int sr, int sc, int d) {
        int r, c;
        int[][] visited = new int[24][24];
        for (r = 1; r <= N; r++) {
            for (c = 1; c <= M; c++) {
                visited[r][c] = 0;
            }
        }
        Queue<qinfo> Q = new LinkedList<>();
        Q.add(new qinfo(sr + dr[d], sc + dc[d], 5));
        visited[sr + dr[d]][sc + dc[d]] = 1;
        while (!Q.isEmpty()) {
            int nr, nc;
            qinfo now = Q.poll();
            r = now.r;
            c = now.c;
            B[r][c] += now.t;
            if (now.t == 1) {
                continue;
            }
            if (d == DIR.RIGHT.getValue()) {
                nr = r - 1;
                nc = c + 1;
                if (in_range(nr, nc) && visited[nr][nc] == 0 && (W[r][c] & DIR_MASK.U_MASK.getValue()) == 0 && (W[nr][nc] & DIR_MASK.L_MASK.getValue()) == 0) {
                    Q.add(new qinfo(nr, nc, now.t - 1));
                    visited[nr][nc] = 1;
                }
                nr = r;
                if (in_range(nr, nc) && visited[nr][nc] == 0 && (W[r][c] & DIR_MASK.R_MASK.getValue()) == 0) {
                    Q.add(new qinfo(nr, nc, now.t - 1));
                    visited[nr][nc] = 1;
                }
                nr = r + 1;
                if (in_range(nr, nc) && visited[nr][nc] == 0 && (W[r][c] & DIR_MASK.D_MASK.getValue()) == 0 && (W[nr][nc] & DIR_MASK.L_MASK.getValue()) == 0) {
                    Q.add(new qinfo(nr, nc, now.t - 1));
                    visited[nr][nc] = 1;
                }
            } else if (d == DIR.LEFT.getValue()) {
                nr = r - 1;
                nc = c - 1;
                if (in_range(nr, nc) && visited[nr][nc] == 0 && (W[r][c] & DIR_MASK.U_MASK.getValue()) == 0 && (W[nr][nc] & DIR_MASK.R_MASK.getValue()) == 0) {
                    Q.add(new qinfo(nr, nc, now.t - 1));
                    visited[nr][nc] = 1;
                }
                nr = r;
                if (in_range(nr, nc) && visited[nr][nc] == 0 && (W[r][c] & DIR_MASK.L_MASK.getValue()) == 0) {
                    Q.add(new qinfo(nr, nc, now.t - 1));
                    visited[nr][nc] = 1;
                }
                nr = r + 1;
                if (in_range(nr, nc) && visited[nr][nc] == 0 && (W[r][c] & DIR_MASK.D_MASK.getValue()) == 0 && (W[nr][nc] & DIR_MASK.R_MASK.getValue()) == 0) {
                    Q.add(new qinfo(nr, nc, now.t - 1));
                    visited[nr][nc] = 1;
                }
            } else if (d == DIR.UP.getValue()) {
                nr = r - 1;
                nc = c - 1;
                if (in_range(nr, nc) && visited[nr][nc] == 0 && (W[r][c] & DIR_MASK.L_MASK.getValue()) == 0 && (W[nr][nc] & DIR_MASK.D_MASK.getValue()) == 0) {
                    Q.add(new qinfo(nr, nc, now.t - 1));
                    visited[nr][nc] = 1;
                }
                nc = c;
                if (in_range(nr, nc) && visited[nr][nc] == 0 && (W[r][c] & DIR_MASK.U_MASK.getValue()) == 0) {
                    Q.add(new qinfo(nr, nc, now.t - 1));
                    visited[nr][nc] = 1;
                }
                nc = c + 1;
                if (in_range(nr, nc) && visited[nr][nc] == 0 && (W[r][c] & DIR_MASK.R_MASK.getValue()) == 0 && (W[nr][nc] & DIR_MASK.D_MASK.getValue()) == 0) {
                    Q.add(new qinfo(nr, nc, now.t - 1));
                    visited[nr][nc] = 1;
                }
            } else {
                nr = r + 1;
                nc = c - 1;
                if (in_range(nr, nc) && visited[nr][nc] == 0 && (W[r][c] & DIR_MASK.L_MASK.getValue()) == 0 && (W[nr][nc] & DIR_MASK.U_MASK.getValue()) == 0) {
                    Q.add(new qinfo(nr, nc, now.t - 1));
                    visited[nr][nc] = 1;
                }
                nc = c;
                if (in_range(nr, nc) && visited[nr][nc] == 0 && (W[r][c] & DIR_MASK.D_MASK.getValue()) == 0) {
                    Q.add(new qinfo(nr, nc, now.t - 1));
                    visited[nr][nc] = 1;
                }
                nc = c + 1;
                if (in_range(nr, nc) && visited[nr][nc] == 0 && (W[r][c] & DIR_MASK.R_MASK.getValue()) == 0 && (W[nr][nc] & DIR_MASK.U_MASK.getValue()) == 0) {
                    Q.add(new qinfo(nr, nc, now.t - 1));
                    visited[nr][nc] = 1;
                }
            }
        }
    }

    static void build_B() {
        for (int r = 1; r <= N; r++) {
            for (int c = 1; c <= M; c++) {
                if (0 < C[r][c] && C[r][c] < 5) {
                    build_B_sub(r, c, C[r][c]);
                }
            }
        }
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        N = sc.nextInt();
        M = sc.nextInt();
        K = sc.nextInt();
        A = new int[24][24];
        B = new int[24][24];
        C = new int[24][24];
        W = new int[24][24];
        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= M; j++) {
                C[i][j] = sc.nextInt();
            }
        }
        int w = sc.nextInt();
        while (w-- > 0) {
            int r = sc.nextInt();
            int c = sc.nextInt();
            int t = sc.nextInt();
            if (t == 0) {
                W[r][c] |= DIR_MASK.U_MASK.getValue();
                W[r - 1][c] |= DIR_MASK.D_MASK.getValue();
            } else {
                W[r][c] |= DIR_MASK.R_MASK.getValue();
                W[r][c + 1] |= DIR_MASK.L_MASK.getValue();
            }
        }
        build_B();
        for (int step = 1; step <= 100; step++) {
            for (int r = 1; r <= N; r++) {
                for (int c = 1; c <= M; c++) {
                    A[r][c] += B[r][c];
                }
            }
            int[][] X = new int[24][24];
            for (int r = 1; r <= N; r++) {
                for (int c = 1; c <= M; c++) {
                    X[r][c] = 0;
                }
            }
            for (int r = 1; r <= N; r++) {
                for (int c = 1; c <= M; c++) {
                    int nr, nc, diff;
                    nr = r;
                    nc = c + 1;
                    if (in_range(nr, nc) && (W[r][c] & DIR_MASK.R_MASK.getValue()) == 0) {
                        diff = (A[r][c] - A[nr][nc]) / 4;
                        X[r][c] -= diff;
                        X[nr][nc] += diff;
                    }
                    nr = r + 1;
                    nc = c;
                    if (in_range(nr, nc) && (W[r][c] & DIR_MASK.D_MASK.getValue()) == 0) {
                        diff = (A[r][c] - A[nr][nc]) / 4;
                        X[r][c] -= diff;
                        X[nr][nc] += diff;
                    }
                }
            }
            for (int r = 1; r <= N; r++) {
                for (int c = 1; c <= M; c++) {
                    A[r][c] += X[r][c];
                }
            }
            for (int c = 1; c <= M; c++) {
                if (A[1][c] >= 1) {
                    A[1][c]--;
                }
                if (A[N][c] >= 1) {
                    A[N][c]--;
                }
            }
            for (int r = 2; r < N; r++) {
                if (A[r][1] >= 1) {
                    A[r][1]--;
                }
                if (A[r][M] >= 1) {
                    A[r][M]--;
                }
            }
            int is_ok = 1;
            for (int r = 1; r <= N; r++) {
                for (int c = 1; c <= M; c++) {
                    if (C[r][c] == 5 && A[r][c] < K) {
                        is_ok = 0;
                    }
                }
            }
            if (is_ok == 1) {
                System.out.print(step);
                return;
            }
        }
        System.out.print("101");
    }
}