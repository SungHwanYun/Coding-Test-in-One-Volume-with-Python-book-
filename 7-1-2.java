import java.util.*;

public class Main {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int a = scanner.nextInt();
        int k = scanner.nextInt();
        int[] D = new int[k + 1];
        for (int i = a + 1; i <= k; i++) {
            D[i] = D[i - 1] + 1;
            if (i % 2 == 0 && i / 2 >= a) {
                D[i] = Math.min(D[i], D[i / 2] + 1);
            }
        }
        System.out.println(D[k]);
    }
}
