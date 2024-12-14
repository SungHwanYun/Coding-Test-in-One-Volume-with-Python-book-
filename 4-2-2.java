import java.util.Scanner;

public class Main {
    public static int do_solve(int n) {
        if (n == 1) return 1;
        return n + do_solve(n - 1);
    }

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int n = scanner.nextInt();
        System.out.println(do_solve(n));
    }
}
