import java.util.*;

public class Main {
    static int N, K;
    static int[][] A = new int[100][100];
    static int[][] B = new int[100][100];
    static int[] S = new int[100];
    public static void main(String[] args) {
        Scanner input = new Scanner(System.in);
        N = input.nextInt();
        K = input.nextInt();
        for (int i = 0; i < N; i++) {
            A[i][0] = input.nextInt();
            S[i] = 1;
        }
        int step = 0;
        while (true) {
            if (is_ok()) break;
            add_one_fish();
            build_up();
            adjust_fish();
            spread_fish();
            build_up_half();
            build_up_half();
            adjust_fish();
            spread_fish();
            step++;
        }
        System.out.println(step);
    }

    public static boolean is_ok() {
        int mx = 0;
        int mn = (int)1e9;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < S[i]; j++) {
                mx = Math.max(mx, A[i][j]);
                mn = Math.min(mn, A[i][j]);
            }
        }
        return (mx - mn) <= K;
    }

    public static void add_one_fish() {
        int mn = (int)1e9;
        for (int i = 0; i < N; i++)
            mn = Math.min(mn, A[i][0]);
        for (int i = 0; i < N; i++)
            if (A[i][0] == mn)
                A[i][0]++;
    }

    public static void move_column(int src, int dst) {
        for (int i = 0; i < S[src]; i++)
            A[dst][i] = A[src][i];
        S[dst] = S[src];
    }

    public static void build_up_sub(int e) {
        for (int i = e; i >= 0; i--)
            for (int j = e + 1, k = 0; k < S[i]; j++, k++)
                A[j][S[j]++] = A[i][k];
        for (int i = 0, j = e + 1; j < N; i++, j++)
            move_column(j, i);
        N = N - e - 1;
    }

    public static void build_up() {
        while (true) {
            int i;
            for (i = 1; i < N; i++)
                if (S[i] == 1)
                    break;
            if (S[0] > N - i) break;
            build_up_sub(i - 1);
        }
    }

    public static void adjust_fish_sub(int r1, int c1, int r2, int c2) {
        int diff;
        if (A[r1][c1] > A[r2][c2]) {
            diff = (A[r1][c1] - A[r2][c2]) / 5;
            B[r1][c1] -= diff;
            B[r2][c2] += diff;
        }
        else {
            diff = (A[r2][c2] - A[r1][c1]) / 5;
            B[r1][c1] += diff;
            B[r2][c2] -= diff;
        }
    }

    public static void adjust_fish() {
        for (int r = 0; r < N; r++)
            for (int c = 0; c < S[r]; c++)
                B[r][c] = 0;
        for (int r = 0; r < N; r++) {
            for (int c = 0; c < S[r]; c++) {
                if (r < N - 1 && c < S[r + 1])
                    adjust_fish_sub(r, c, r + 1, c);
                if (c + 1 < S[r])
                    adjust_fish_sub(r, c, r, c + 1);
            }
        }
        for (int r = 0; r < N; r++)
            for (int c = 0; c < S[r]; c++)
                A[r][c] += B[r][c];
    }

    public static void spread_fish() {
        int k = 0;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < S[i]; j++)
                B[k++][0] = A[i][j];
        N = k;
        for (int i = 0; i < N; i++) {
            A[i][0] = B[i][0];
            S[i] = 1;
        }
    }

    public static void build_up_half() {
        int r1, c, r2, e = N / 2 - 1;
        for (c = S[0] - 1; c >= 0; c--)
            for (r1 = e, r2 = e + 1; r1 >= 0; r1--, r2++)
                A[r2][S[r2]++] = A[r1][c];
        for (r1 = 0, r2 = e + 1; r2 < N; r1++, r2++)
            move_column(r2, r1);
        N /= 2;
    }
}