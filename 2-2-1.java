import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        String A = scanner.next();
        String B = "";
        for (int i = 1; i < A.length(); i += 2) {
            B = B + A.charAt(i);
        }
        System.out.println(B);
    }
}
