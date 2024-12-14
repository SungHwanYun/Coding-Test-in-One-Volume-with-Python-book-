import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        String A = scanner.next();
        String B = "";
        for (int i = 0; i < A.length(); i++) {
            char b;
            if ('A' <= A.charAt(i) && A.charAt(i) <= 'Z') {
                b = A.charAt(i);
            } else {
                b = (char) (A.charAt(i) + 'A' - 'a');
            }
            B = B + b;
        }
        System.out.println(B);
    }
}
