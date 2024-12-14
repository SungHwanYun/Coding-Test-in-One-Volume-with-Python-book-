import java.util.Scanner;
import java.util.Vector;

public class Main {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int n = scanner.nextInt();
        Vector<Integer> A = new Vector<>(n);
        for (int i = 0; i < n; i++) {
            A.add(scanner.nextInt());
        }
        int i = scanner.nextInt();
        int j = scanner.nextInt();
        int k = scanner.nextInt();
        for (int idx = i; idx <= j; idx++) {
            A.set(idx, A.get(idx) * k);
        }
        int answer = 0;
        for (int num : A) {
            answer += num;
        }
        System.out.println(answer);
    }
}
