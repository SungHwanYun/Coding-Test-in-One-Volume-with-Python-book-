import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        String s = scanner.next();
        int k = scanner.nextInt();
        scanner.close();
        
        while (s.length() < k) {
            s += s.charAt(s.length() - 1);
        }
        
        System.out.println(s);
    }
}
