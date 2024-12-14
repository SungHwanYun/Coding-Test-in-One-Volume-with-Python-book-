import java.util.*;

public class Main {
    static int N;

    public static int P(List<Integer> X) {
        int ret = 1;
        for (int x : X) {
            ret *= x;
        }
        return ret;
    }

    public static int getJoinedNum(List<Integer> X) {
        StringBuilder x = new StringBuilder();
        for (int num : X) {
            x.append(num);
        }
        return Integer.parseInt(x.toString());
    }

    public static List<Integer> solve(int n, List<Integer> A, List<Integer> B) {
        List<Integer> ret = new ArrayList<>();
        ret.add(-1);
        if (n == 0) {
            int pa = P(A), pb = P(B);
            if (pa < pb) {
                return B;
            }
            return ret;
        }
        int start = 1;
        if (B.size() != 0) {
            start = B.get(B.size() - 1);
        }
        for (int card = start; card <= 9; card++) {
            B.add(card);
            List<Integer> ret2 = solve(n - 1, A, B);
            if (ret2.get(0) != -1) {
                if (ret.get(0) == -1) {
                    ret = new ArrayList<>(Collections.nCopies(N, 0));
                    Collections.copy(ret, ret2);
                } else {
                    int retNum = getJoinedNum(ret);
                    int ret2Num = getJoinedNum(ret2);
                    if (retNum > ret2Num) {
                        Collections.copy(ret, ret2);
                    }
                }
            }
            B.remove(B.size() - 1);
        }
        return ret;
    }

    public static List<Integer> solution(int n, List<Integer> A) {
        Collections.sort(A);
        List<Integer> B = new ArrayList<>();
        return solve(n, A, B);
    }

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int n = scanner.nextInt();
        N = n;
        List<Integer> A = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            A.add(scanner.nextInt());
        }
        List<Integer> B = solution(n, A);
        for (int b : B) {
            System.out.print(b + " ");
        }
    }
}
