import java.util.Scanner;
import java.util.ArrayList;
import java.util.List;

class Main {
    public static void do_add_query(List<List<Long>> A, int i1, int j1, int i2, int j2, int k) {
        A.get(i1).set(j1, A.get(i1).get(j1) + k);
        if (j2 + 1 < A.size()) {
            A.get(i1).set(j2 + 1, A.get(i1).get(j2 + 1) - k);
        }
        if (i2 + 1 < A.size()) {
            A.get(i2 + 1).set(j1, A.get(i2 + 1).get(j1) - k);
        }
        if (i2 + 1 < A.size() && j2 + 1 < A.size()) {
            A.get(i2 + 1).set(j2 + 1, A.get(i2 + 1).get(j2 + 1) + k);
        }
    }

    public static long get_sum(List<List<Long>> A, int i1, int j1, int i2, int j2) {
        long ret = 0;
        for (int i = i1; i <= i2; i++) {
            for (int j = j1; j <= j2; j++) {
                ret += A.get(i).get(j);
            }
        }
        return ret;
    }

    public static void solution(int n, List<List<Long>> A, int m, List<List<Integer>> Q) {
        List<List<Long>> psum = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            List<Long> row = new ArrayList<>();
            for (int j = 0; j < n; j++) {
                row.add(0L);
            }
            psum.add(row);
        }
        for (List<Integer> q : Q) {
            if (q.get(0) == 1) {
                do_add_query(psum, q.get(1), q.get(2), q.get(3), q.get(4), q.get(5));
            } else {
                for (int r = 0; r < n; r++) {
                    for (int c = 1; c < n; c++) {
                        psum.get(r).set(c, psum.get(r).get(c) + psum.get(r).get(c - 1));
                    }
                }
                for (int c = 0; c < n; c++) {
                    for (int r = 1; r < n; r++) {
                        psum.get(r).set(c, psum.get(r).get(c) + psum.get(r - 1).get(c));
                    }
                }
                System.out.println(get_sum(A, q.get(1), q.get(2), q.get(3), q.get(4)) + get_sum(psum, q.get(1), q.get(2), q.get(3), q.get(4)));
            }
        }
    }

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int n = scanner.nextInt();
        int m = scanner.nextInt();
        List<List<Long>> A = new ArrayList<>();
        List<List<Integer>> Q = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            List<Long> row = new ArrayList<>();
            for (int j = 0; j < n; j++) {
                row.add(scanner.nextLong());
            }
            A.add(row);
        }
        for (int i = 0; i < m; i++) {
            List<Integer> query = new ArrayList<>();
            int a = scanner.nextInt(); query.add(a);
            int t;
            if (a==1) t=6;
            else t=5;
            for (int j = 1; j < t; j++) {
                query.add(scanner.nextInt());
            }
            Q.add(query);
        }
        solution(n, A, m, Q);
    }
}
