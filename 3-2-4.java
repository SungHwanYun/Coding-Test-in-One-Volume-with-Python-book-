import java.util.Scanner;

public class Main {
    public static void do_add_query(int[] T, int i, int j) {
        for (int t = i; t < j; t++)
            T[t]++;
    }

    public static int translate_time(String t) {
        int x = Integer.parseInt(t.substring(0, 2)) * 60 + Integer.parseInt(t.substring(3, 5));
        assert (0 <= x && x < 3600);
        return x;
    }

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int[] T = new int[3600];
        int n = scanner.nextInt();
        while (n-- > 0) {
            int op = scanner.nextInt();
            if (op == 1) {
                String x = scanner.next();
                String y = scanner.next();
                do_add_query(T, translate_time(x), translate_time(y));
            } else {
                String x = scanner.next();
                System.out.println(T[translate_time(x)]);
            }
        }
    }
}
