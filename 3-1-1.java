import java.util.Scanner;
import java.util.ArrayList;

public class Main {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int n = scanner.nextInt();
        int m = scanner.nextInt();
        ArrayList<Long> A = new ArrayList<Long>();
        for (int i = 0; i < n; i++) {
            A.add(scanner.nextLong());
        }
        for (int i = 0; i < m; i++) {
            long k = scanner.nextLong();
            int cnt = 0;
            for (int j = 0; j < n; j++) {
                if (A.get(j) >= k) {
                    cnt++;
                }
            }
            System.out.println(cnt);
        }
    }
}
