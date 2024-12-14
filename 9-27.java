import java.util.*;

public class Main {
    static int[] dr = {0, 0, -1, 1};
    static int[] dc = {-1, 1, 0, 0};
    static int N, M;
    static int[][] A = new int[21][21];
    static int[][] B = new int[21][21];
    static int[][] visited = new int[21][21];

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        N = sc.nextInt();
        M = sc.nextInt();
        for (int r = 1; r <= N; r++) {
            for (int c = 1; c <= N; c++) {
                A[r][c] = sc.nextInt();
            }
        }
        int ans = 0;
        while (true) {
            int ret = find_erase_bg();
            if (ret == 0) break;
            ans += ret;
            apply_gravity();
            rotate_blocks();
            apply_gravity();
        }
        System.out.println(ans);
    }

    static boolean in_range(int r, int c) {
        return 1 <= r && r <= N && 1 <= c && c <= N;
    }

    static ArrayList<Integer> get_size(int r, int c, int color) {
        visited[r][c] = 1;
        ArrayList<Integer> ret = new ArrayList<>();
        ret.add(0);
        ret.add(0);
        if (A[r][c] == 0) ret.set(0, ret.get(0) + 1);
        else ret.set(1, ret.get(1) + 1);
        for (int i = 0; i < 4; i++) {
            int nr = r + dr[i];
            int nc = c + dc[i];
            if (in_range(nr, nc) == false) continue;
            if (visited[nr][nc] == 1) continue;
            if (A[nr][nc] == -1 || A[nr][nc] == -2) continue;
            if (A[nr][nc] != 0 && A[nr][nc] != color) continue;
            ArrayList<Integer> v = get_size(nr, nc, color);
            ret.set(0, ret.get(0) + v.get(0));
            ret.set(1, ret.get(1) + v.get(1));
        }
        return ret;
    }

    static void fill_zero(int r, int c, int color) {
        A[r][c] = -2;
        for (int i = 0; i < 4; i++) {
            int nr = r + dr[i];
            int nc = c + dc[i];
            if (in_range(nr, nc) == false) continue;
            if (A[nr][nc] == -1 || A[nr][nc] == -2) continue;
            if (A[nr][nc] != 0 && A[nr][nc] != color) continue;
            fill_zero(nr, nc, color);
        }
    }

    static void reset_visited() {
        for (int r = 1; r <= N; r++) {
            for (int c = 1; c <= N; c++) {
                if (A[r][c] == 0) visited[r][c] = 0;
            }
        }
    }

    static int find_erase_bg() {
        int mx_r = -1, mx_c = -1;
        int[] mx = {-1, -1};
        int r, c;
        for (r = 1; r <= N; r++) {
            for (c = 1; c <= N; c++) {
                visited[r][c] = 0;
            }
        }
        for (r = 1; r <= N; r++) {
            for (c = 1; c <= N; c++) {
                if (A[r][c] == -1 || A[r][c] == -2 || A[r][c] == 0) continue;
                reset_visited();
                ArrayList<Integer> ret = get_size(r, c, A[r][c]);
                if (ret.get(0) + ret.get(1) < 2) continue;
                if (mx[0] + mx[1] < ret.get(0) + ret.get(1) || (mx[0] + mx[1] == ret.get(0) + ret.get(1) && mx[0] <= ret.get(0))) {
                    mx_r = r;
                    mx_c = c;
                    mx[0] = ret.get(0);
                    mx[1] = ret.get(1);
                }
            }
        }
        if (mx_r == -1) return 0;
        fill_zero(mx_r, mx_c, A[mx_r][mx_c]);
        return (mx[0] + mx[1]) * (mx[0] + mx[1]);
    }

    static void apply_gravity() {
        int r, c, x;
        for (c = 1; c <= N; c++) {
            for (r = N - 1; r >= 1; r--) {
                if (A[r][c] == -1 || A[r][c] == -2) continue;
                for (x = r + 1; x <= N; x++) {
                    if (A[x][c] != -2) break;
                }
                x--;
                if (r != x) {
                    A[x][c] = A[r][c];
                    A[r][c] = -2;
                }
            }
        }
    }

    static void rotate_blocks() {
        int r, c;
        for (r = 1; r <= N; r++) {
            for (c = 1; c <= N; c++) {
                B[N - c + 1][r] = A[r][c];
            }
        }
        for (r = 1; r <= N; r++) {
            for (c = 1; c <= N; c++) {
                A[r][c] = B[r][c];
            }
        }
    }
}
