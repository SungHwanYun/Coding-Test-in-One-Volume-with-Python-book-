import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        String S = scanner.next();
        String T = "";
        int i = 0;
        while (i < S.length()) {
            if (S.charAt(i) != 'a' && S.charAt(i) != 'A') {
                T = T + S.charAt(i);
                i++;
                continue;
            }
            int j = i + 1;
            while (j < S.length()) {
                if (S.charAt(j) != 'a' && S.charAt(j) != 'A') {
                    break;
                }
                j++;
            }
            if (j - i == 1) {
                T = T + S.charAt(i);
            } else {
                T = T + 'a';
            }
            i = j;
        }
        System.out.println(T);
    }
}
