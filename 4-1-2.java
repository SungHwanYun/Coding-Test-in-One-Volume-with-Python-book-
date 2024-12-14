import java.util.Scanner;

public class Main {
    public static int pow(int x, int y) {
        int ret = 1;
        while (y-- > 0) {
            ret *= x;
        }
        return ret;
    }

    public static int is_ok(int a) {
        int p = a % 10;
        a /= 10;
        if (p == 0) return 0;
        while (a != 0) {
            int c = a % 10;
            a /= 10;
            if (c == 0 || Math.abs(p - c) > 2) return 0;
            p = c;
        }
        return 1;
    }

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int n = scanner.nextInt();
        int answer = 0;
        for (int i = pow(10, n - 1); i < pow(10, n); i++) {
            if (is_ok(i) == 1) answer++;
        }
        System.out.println(answer);
    }
}
