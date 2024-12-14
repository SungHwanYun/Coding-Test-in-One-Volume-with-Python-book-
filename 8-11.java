import java.util.Scanner;
import java.util.ArrayList;
import java.util.List;

public class Main {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int n = scanner.nextInt();
        int m = scanner.nextInt();
        List<List<Long>> A = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            List<Long> row = new ArrayList<>();
            for (int j = 0; j < n; j++) {
                row.add(scanner.nextLong());
            }
            A.add(row);
        }
        List<List<Integer>> Q = new ArrayList<>();
        for (int i = 0; i < m; i++) {
            List<Integer> query = new ArrayList<>();
            query.add(scanner.nextInt());
            int t = 0;
            if (query.get(0) == 1) {
                t = 6;
            } else {
                t = 5;
            }
            for (int j = 1; j < t; j++) {
                query.add(scanner.nextInt());
            }
            Q.add(query);
        }
        long[][] psum = new long[n][n];
        boolean psum_flag = false;
        for (int q = 0; q < m; q++) {
            if (Q.get(q).get(0) == 1) {
                int i1 = Q.get(q).get(1);
                int j1 = Q.get(q).get(2);
                int i2 = Q.get(q).get(3);
                int j2 = Q.get(q).get(4);
                int k = Q.get(q).get(5);
                psum[i1][j1] += k;
                if (j2 + 1 < n) {
                    psum[i1][j2+1] -= k;
                }
                if (i2 + 1 < n) {
                    psum[i2+1][j1] -= k;
                }
                if (i2 + 1 < n && j2 + 1 < n) {
                    psum[i2+1][j2+1] += k;
                }
            } else {
                if (!psum_flag) {
                    psum_flag = true;
                    for (int r = 0; r < n; r++) {
                        for (int c = 1; c < n; c++) {
                            psum[r][c] += psum[r][c - 1];
                        }
                    }
                    for (int c = 0; c < n; c++) {
                        for (int r = 1; r < n; r++) {
                            psum[r][c] += psum[r - 1][c];
                        }
                    }
                    for (int r = 0; r < n; r++) {
                        for (int c = 0; c < n; c++) {
                            A.get(r).set(c, A.get(r).get(c)+psum[r][c]);
                        }
                    }
                    psum[0][0] = A.get(0).get(0);
                    for (int c = 1; c < n; c++) {
                        psum[0][c] = psum[0][c - 1] + A.get(0).get(c);
                    }
                    for (int r = 1; r < n; r++) {
                        psum[r][0] = psum[r - 1][0] + A.get(r).get(0);
                    }
                    for (int r = 1; r < n; r++) {
                        for (int c = 1; c < n; c++) {
                            psum[r][c] = psum[r - 1][c] + psum[r][c - 1] - psum[r - 1][c - 1] + A.get(r).get(c);
                        }
                    }
                }
                int i1 = Q.get(q).get(1);
                int j1 = Q.get(q).get(2);
                int i2 = Q.get(q).get(3);
                int j2 = Q.get(q).get(4);
                long ret = psum[i2][j2];
                if (i1 > 0) {
                    ret -= psum[i1 - 1][j2];
                }
                if (j1 > 0) {
                    ret -= psum[i2][j1 - 1];
                }
                if (i1 > 0 && j1 > 0) {
                    ret += psum[i1 - 1][j1 - 1];
                }
                System.out.println(ret);
            }
        }
    }
}
