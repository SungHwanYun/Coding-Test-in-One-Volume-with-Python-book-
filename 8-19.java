import java.util.*;

public class Main {
    public static int solution(int n, int k, List<Integer> A) {
        List<Integer> B = new ArrayList<>(n + 1);
        for (int i = 0; i <= n; i++) {
            B.add(0);
        }
        for (int a : A) {
            B.set(a, 1);
        }
        List<Integer> D = new ArrayList<>(n + 2);
        for (int i = 0; i <= n + 1; i++) {
            D.add(0);
        }
        for (int i = n; i > 0; i--) {
            List<Integer> nxt = new ArrayList<>();
            for (int j = i; j <= i + k - 1; j++) {
                if (j > n) break;
                if (B.get(j) == 1) continue;
                nxt.add(D.get(j + 1));
            }
            if (nxt.size() == 0) {
                D.set(i, 0);
                continue;
            }
            Collections.sort(nxt);
            if (nxt.get(0) > 0) {
                D.set(i, -(nxt.get(nxt.size() - 1) + 1));
                continue;
            }
            int ret = 0;
            for (int p = 0; p < nxt.size(); p++) {
                if (nxt.get(p) <= 0) {
                    ret = nxt.get(p);
                }
            }
            D.set(i, -ret + 1);
        }
        return Math.abs(D.get(1));
    }

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int n = scanner.nextInt();
        int k = scanner.nextInt();
        List<Integer> A = new ArrayList<>();
        do {
            int a = scanner.nextInt();
            A.add(a);
        } while (scanner.hasNextInt());
        System.out.println(solution(n, k, A));
    }
}
