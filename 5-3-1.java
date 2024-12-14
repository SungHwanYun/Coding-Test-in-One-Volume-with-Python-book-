import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int n = scanner.nextInt();
        int k = scanner.nextInt();
        long b = 0;
        while (n > 0) {
            int d = n % k;
            n /= k;
            b = b * k + d;
        }
        System.out.println(b);
    }
}
