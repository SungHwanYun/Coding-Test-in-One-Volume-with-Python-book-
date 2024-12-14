import java.util.Scanner;
import java.util.Vector;

public class Main {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int n = scanner.nextInt();
        int m = scanner.nextInt();
        Vector<Integer> A = new Vector<>(n);
        for (int i = 0; i < n; i++) {
            A.add(scanner.nextInt());
        }
        for (int i = 0; i < m; i++) {
            int op = scanner.nextInt();
            if (op == 1) {
                int x = scanner.nextInt();
                int y = scanner.nextInt();
                int z = scanner.nextInt();
                assert (0 <= x && x <= y && y < n);
                for (int j = x; j <= y; j++) {
                    A.set(j, A.get(j) + z);
                }
            } else {
                long answer = 0;
                int x = scanner.nextInt();
                int y = scanner.nextInt();
                assert (0 <= x && x <= y && y < n);
                for (int j = x; j <= y; j++) {
                    answer += A.get(j);
                }
                System.out.println(answer);
            }
        }
    }
}
