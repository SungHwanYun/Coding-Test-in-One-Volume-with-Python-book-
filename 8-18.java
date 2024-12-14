import java.util.*;

public class Main {
    public static int solution(int n, int k, ArrayList<Integer> A) {
        ArrayList<Integer> B = new ArrayList<>(Collections.nCopies(n + 1, 0));
        for (int a : A) {
            B.set(a, 1);
        }
        ArrayList<Integer> D = new ArrayList<>(Collections.nCopies(n + 2, 0));
        for (int i = n; i > 0; i--) {
            for (int j = i; j <= i + k - 1; j++) {
                if (j > n) break;
                if (B.get(j) == 1) {
                    continue;
                }
                if (D.get(j + 1) == 0) {
                    D.set(i, 1);
                    break;
                }
            }
        }
        return D.get(1);
    }

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int n = scanner.nextInt();
        int k = scanner.nextInt();
        ArrayList<Integer> A = new ArrayList<>();
        do {
            int a = scanner.nextInt();
            A.add(a);
        } while (scanner.hasNextInt());
        System.out.println(solution(n, k, A));
    }
}