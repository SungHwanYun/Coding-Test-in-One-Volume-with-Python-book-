import java.util.Scanner;

public class Main {
    public static void doSolve(String A, int B) {
        if (A.length() > 0 && !A.equals("0")) {
            if (A.charAt(A.length() - 1) != '0' || B == 1) {
                System.out.print(A.charAt(A.length() - 1));
                B = 1;
            }
            A = A.substring(0, A.length() - 1);
            doSolve(A, B);
        }
    }

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        String A = scanner.next();
        doSolve(A, 0);
    }
}
