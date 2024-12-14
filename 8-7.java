import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Scanner;

public class Main {
    public static int solve(int n, int k, int a, List<Integer> B) {
        if (a == n) {
            return 0;
        }
        List<Integer> nxt = new ArrayList<>();
        for (int b = a + 1; b <= a + k; b++) {
            if (b > n) {
                break;
            }
            if (B.get(b) == 1) {
                continue;
            }
            nxt.add(solve(n, k, b, B));
        }
        if (nxt.size() == 0) {
            return 0;
        }
        Collections.sort(nxt);
        if (nxt.get(0) > 0) {
            return -(nxt.get(nxt.size() - 1) + 1);
        }
        int ret = 0;
        for (int i = 0; i < nxt.size(); i++) {
            if (nxt.get(i) <= 0) {
                ret = nxt.get(i);
            }
        }
        return -ret + 1;
    }

    public static int solution(int n, int k, List<Integer> A) {
        List<Integer> B = new ArrayList<>(Collections.nCopies(n + 1, 0));
        for (int a : A) {
            B.set(a, 1);
        }
        return Math.abs(solve(n, k, 0, B));
    }

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int n = scanner.nextInt();
        int k = scanner.nextInt();
        List<Integer> A = new ArrayList<>();
        do {
            A.add(scanner.nextInt());
        } while (scanner.hasNext());
        System.out.println(solution(n, k, A));
    }
}
