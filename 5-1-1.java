import java.util.*;

public class Main {
    public static void do_add_query(int[] T, int i, int j) {
        T[i]++;
        T[j]--;
    }

    public static int translate_time(String t) {
        int x = Integer.parseInt(t.substring(0, 2)) * 3600 + Integer.parseInt(t.substring(3, 5)) * 60 + Integer.parseInt(t.substring(6, 8));
        return x;
    }

    public static long get_sum(int[] T, int i, int j) {
        for (int t = 1; t < 24 * 60 * 60; t++)
            T[t] += T[t - 1];
        long ret = 0;
        for (int t = i; t < j; t++)
            ret += T[t];
        return ret;
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int[] T = new int[86400];
        int n = sc.nextInt();
        while (n-- > 0) {
            int op = sc.nextInt();
            if (op == 1) {
                String x = sc.next();
                String y = sc.next();
                do_add_query(T, translate_time(x), translate_time(y));
            } else {
                String x = sc.next();
                String y = sc.next();
                System.out.println(get_sum(T, translate_time(x), translate_time(y)));
            }
        }
    }
}
