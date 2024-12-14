import java.util.Scanner;
import java.util.ArrayList;
import java.util.List;

public class Main {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        String A = scanner.next();
        List<Character> B = new ArrayList<>();

        do {
            char b = scanner.next().charAt(0);
            B.add(b);
        } while (scanner.hasNext());
        
        for (int i = 0; i < B.size(); i++) {
            char b = B.get(i);
            for (int j = 0; j < A.length(); j++) {
                if (A.charAt(j) == b) {
                    A = A.substring(0, j) + (char)(b + 'a' - 'A') + A.substring(j + 1);
                }
            }
        }
        System.out.println(A);
    }
}
