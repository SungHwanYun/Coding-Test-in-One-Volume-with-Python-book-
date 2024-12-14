import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int n = scanner.nextInt();
        int k = scanner.nextInt();
        long a = 0;
        while (n > 0) {
            int d = n % k;
            n /= k;
            a = a + d;
        }
        String b = "";
        while (a > 0) {
            int d = (int) (a % k);
            a = a / k;
            b = Integer.toString(d) + b;
        }
        System.out.println(b);
    }
}
