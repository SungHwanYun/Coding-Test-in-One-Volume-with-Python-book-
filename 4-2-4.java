import java.util.*;

public class Main {
    public static int do_solve(ArrayList<Integer> A, int n) {
        if (A.size() == n) return 1;
        int s, e;
        if (A.size() == 0) {
            s = 1; e = 9;
        }
        else {
            s = Math.max(A.get(A.size() - 1) - 2, 1);
            e = Math.min(A.get(A.size() - 1) + 2, 9);
        }
        int ret = 0;
        for (int i = s; i <= e; i++) {
            A.add(i);
            ret += do_solve(A, n);
            A.remove(A.size() - 1);
        }
        return ret;
    }
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int n = scanner.nextInt();
        ArrayList<Integer> A = new ArrayList<>();
        System.out.println(do_solve(A, n));
    }
}
