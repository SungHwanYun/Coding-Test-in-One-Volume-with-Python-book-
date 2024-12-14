import java.util.Scanner;
import java.util.Vector;

public class Main {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int n = scanner.nextInt();
        Vector<Vector<Integer>> A = new Vector<>(n);
        for (int i = 0; i < n; i++) {
            Vector<Integer> row = new Vector<>(n);
            for (int j = 0; j < n; j++) {
                row.add(scanner.nextInt());
            }
            A.add(row);
        }
        int i1 = scanner.nextInt();
        int j1 = scanner.nextInt();
        int i2 = scanner.nextInt();
        int j2 = scanner.nextInt();
        int k = scanner.nextInt();
        for (int i = i1; i <= i2; i++) {
            for (int j = j1; j <= j2; j++) {
                A.get(i).set(j, A.get(i).get(j) * k);
            }
        }
        int answer = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                answer += A.get(i).get(j);
            }
        }
        System.out.println(answer);
    }
}
