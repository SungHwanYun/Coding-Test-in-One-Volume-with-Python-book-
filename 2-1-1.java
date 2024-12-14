import java.util.Scanner;
import java.util.ArrayList;

public class Main {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int n = scanner.nextInt();
        int k = scanner.nextInt();
        ArrayList<Integer> A = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            A.add(scanner.nextInt());
        }
        int answer = 0;
        for (int i = 0; i < n; i++) {
            if (A.get(i) == k) {
                answer++;
            }
        }
        System.out.println(answer);
    }
}
