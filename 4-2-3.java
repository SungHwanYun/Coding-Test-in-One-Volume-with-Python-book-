import java.util.*;

public class Main {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int n = scanner.nextInt();
        System.out.println(doSolve(n));
    }

    public static int doSolve(int n) {
        if (n <= 2) return 1;
        return doSolve(n - 1) + doSolve(n - 2);
    }
}
