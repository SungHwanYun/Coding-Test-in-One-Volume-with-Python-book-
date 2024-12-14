import java.util.*;

public class Main {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();
        List<Integer> D = new ArrayList<>(Collections.nCopies(n + 1, 1));
        for (int i = 3; i <= n; i++) {
            D.set(i, (D.get(i - 1) + D.get(i - 2)) % 987654321);
        }
        System.out.println(D.get(n));
    }
}
