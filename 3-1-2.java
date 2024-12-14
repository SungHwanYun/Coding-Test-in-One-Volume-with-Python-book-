import java.util.Scanner;
import java.util.ArrayList;

public class Main {
    public static void main(String[] args) {
        Scanner input = new Scanner(System.in);
        int n = input.nextInt();
        int m = input.nextInt();
        ArrayList<String> A = new ArrayList<String>();
        for (int i = 0; i < n; i++) {
            A.add(input.next());
        }
        for (int i = 0; i < m; i++) {
            String s = input.next();
            if (s.equals("-")) {
                System.out.println(n);
                continue;
            }
            int cnt = 0;
            for (int j = 0; j < n; j++) {
                if (A.get(j).equals(s)) {
                    cnt++;
                }
            }
            System.out.println(cnt);
        }
    }
}
