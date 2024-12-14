import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        long answer = 0;
        do {
            long a = scanner.nextLong();
            if (isPrime(a) == 1)
                answer += a;
        } while (scanner.hasNext());
        System.out.println(answer);
    }

    public static int isPrime(long a) {
        if (a < 2) return 0;
        for (long i = 2; i * i <= a; i++) {
            if (a % i == 0) return 0;
        }
        return 1;
    }
}
