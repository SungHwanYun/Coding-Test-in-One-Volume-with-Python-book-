import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        String A = scanner.next();
        String B = "";
        for (int i = 0; i < A.length(); i++) {
            if ('a' <= A.charAt(i) && A.charAt(i) <= 'z') {
                B = B + A.charAt(i);
            }
        }
        System.out.println(B);
    }
}
