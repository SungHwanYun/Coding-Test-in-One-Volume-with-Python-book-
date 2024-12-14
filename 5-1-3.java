import java.util.Scanner;

public class Main {
    public static void do_add_query(int[] psum, int i, int j, int k, int n) {
        psum[i] += k;
        if (j + 1 < n) psum[j + 1] -= k;
    }

    public static long get_sum(long[] psum2, int i, int j) {
        long ret = psum2[j];
        if (i > 0) ret -= psum2[i - 1];
        return ret;
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();
        int m = sc.nextInt();
        int[] A = new int[100004];
        int[] psum = new int[100004];
        long[] psum2 = new long[100004];
        for (int i = 0; i < n; i++) A[i] = sc.nextInt();
        int psum_flag = 0;
        while (m-- > 0) {
            int op = sc.nextInt();
            int i = sc.nextInt();
            int j = sc.nextInt();
            if (op == 1) {
                int k = sc.nextInt();
                do_add_query(psum, i, j, k, n);
            } else {
                if (psum_flag == 0) {
                    psum_flag = 1;
                    for (int t = 1; t < n; t++) psum[t] += psum[t - 1];
                    for (int t = 0; t < n; t++) A[t] += psum[t];
                    psum2[0] = A[0];
                    for (int t = 1; t < n; t++) psum2[t] = psum2[t - 1] + A[t];
                }
                System.out.println(get_sum(psum2, i, j));
            }
        }
    }
}
