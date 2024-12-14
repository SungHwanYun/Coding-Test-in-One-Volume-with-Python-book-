import java.util.*;

public class Main {
    public static int solve(int n, int k, int a, ArrayList<Integer> B) {
        if (a == n) {
            return 0;
        }
        for (int b = a + 1; b <= a + k; b++) {
            if (b > n) {
                break;
            }
            if (B.get(b) == 1) {
                continue;
            }
            if (solve(n, k, b, B) == 0) {
                return 1;
            }
        }
        return 0;
    }

    public static int solution(int n, int k, ArrayList<Integer> A) {
        ArrayList<Integer> B = new ArrayList<>(Collections.nCopies(n + 1, 0));
        for (int a : A) {
            B.set(a, 1);
        }
        return solve(n, k, 0, B);
    }

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int n = scanner.nextInt();
        int k = scanner.nextInt();
        ArrayList<Integer> A = new ArrayList<>();
        do {
            A.add(scanner.nextInt());
        } while (scanner.hasNext());
        System.out.println(solution(n, k, A));
    }
}
