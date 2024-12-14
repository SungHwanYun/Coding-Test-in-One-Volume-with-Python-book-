import java.util.Scanner;

public class Main {
    public static void do_add_query(int[] psum, int i, int j, int k, int n) {
        psum[i] += k;
        if (j + 1 < n) psum[j + 1] -= k;
    }

    public static long get_sum(int[] A, int[] psum, int i, int j, int n) {
        for (int t = 1; t < n; t++) psum[t] += psum[t - 1];
        long ret = 0;
        for (int t = i; t <= j; t++) ret += psum[t] + A[t];
        return ret;
    }

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int n = scanner.nextInt();
        int m = scanner.nextInt();
        int[] A = new int[100004];
        for (int i = 0; i < n; i++) A[i] = scanner.nextInt();
        int[] psum = new int[100004];
        while (m-- > 0) {
            int op = scanner.nextInt();
            if (op == 1) {
                int i = scanner.nextInt();
                int j = scanner.nextInt();
                int k = scanner.nextInt();
                do_add_query(psum, i, j, k, n);
            } else {
                int i = scanner.nextInt();
                int j = scanner.nextInt();
                System.out.println(get_sum(A, psum, i, j, n));
            }
        }
    }
}
