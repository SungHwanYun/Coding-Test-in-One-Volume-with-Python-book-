import java.util.Scanner;

public class Main {
    static enum DIR {
        LEFT, DOWN, RIGHT, UP
    }

    static int[] dr = {0, 1, 0, -1};
    static int[] dc = {-1, 0, 1, 0};
    static int N, M;
    static int[][] A = new int[50][50];
    static int[][] B = new int[50][50];
    static int[] C = new int[2500];
    static int[] D = new int[2500];
    static int[] ans = new int[4];

    static boolean in_range(int r, int c) {
        return 0 <= r && r < N && 0 <= c && c < N;
    }

    static void build_BC() {
        int r, c, d, remain_move, total_move, step, cnt = 0;
        r = c = N / 2;
        d = DIR.LEFT.ordinal();
        remain_move = total_move = 1;
        step = 1;
        while (true) {
            if (!in_range(r, c)) break;
            B[r][c] = cnt;
            C[cnt] = A[r][c];
            cnt++;
            r = r + dr[d];
            c = c + dc[d];
            remain_move--;
            if (remain_move == 0 && step == 2) {
                total_move++;
                remain_move = total_move;
                step = 1;
                d = (d + 1) % 4;
            } else if (remain_move == 0) {
                remain_move = total_move;
                step = 2;
                d = (d + 1) % 4;
            }
        }
    }

    static void move_thing_slow() {
        int i, j, empty = 0;
        for (i = 1; i < N * N; i++) {
            if (C[i] == 0) {
                empty++;
                continue;
            }
            if (empty == 0) continue;
            for (j = 1; j < i; j++)
                if (C[j] == 0) break;
            C[j] = C[i];
            C[i] = 0;
        }
    }

    static void move_thing_fast() {
        int i, j;
        for (i = 1, j = 1; i < N * N; i++) {
            if (C[i] != 0) {
                if (i == j) {
                    j++;
                } else {
                    C[j] = C[i];
                    j++;
                }
            }
        }
    }

    static int explode_thing_sub() {
        int i = 1, j, ret = 0;
        while (i < N * N) {
            if (C[i] == 0) {
                i++;
                continue;
            }
            for (j = i + 1; j < N * N; j++) {
                if (C[i] != C[j]) break;
            }
            if (j - i >= 4) {
                ans[C[i]] += j - i;
                for (int k = i; k < j; k++) C[k] = 0;
                ret = 1;
            }
            i = j;
        }
        return ret;
    }

    static void explode_thing() {
        while (true) {
            int ret = explode_thing_sub();
            if (ret == 0) break;
            move_thing_slow();
        }
    }

    static void change_thing() {
        int i = 1, j, k = 1;
        while (i < N * N) {
            if (C[i] == 0) {
                i++;
                continue;
            }
            if (k >= N * N - 1) break;
            for (j = i + 1; j < N * N; j++) {
                if (C[i] != C[j]) break;
            }
            D[k++] = j - i;
            D[k++] = C[i];
            i = j;
        }
        for (i = 0; i < k; i++)
            C[i] = D[i];
        for (i = k; i < N * N; i++)
            C[i] = 0;
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int r, c, d, s;
        N = sc.nextInt();
        M = sc.nextInt();
        for (r = 0; r < N; r++)
            for (c = 0; c < N; c++)
                A[r][c] = sc.nextInt();
        build_BC();
        while (M-- > 0) {
            d = sc.nextInt();
            s = sc.nextInt();
            int[] rr = {0, -1, 1, 0, 0};
            int[] cc = {0, 0, 0, -1, 1};
            r = c = N / 2;
            while (s-- > 0) {
                r = r + rr[d];
                c = c + cc[d];
                C[B[r][c]] = 0;
            }
            move_thing_slow();
            explode_thing();
            change_thing();
        }
        System.out.println(ans[1] + 2 * ans[2] + 3 * ans[3]);
    }
}