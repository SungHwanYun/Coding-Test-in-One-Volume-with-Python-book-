import java.util.*;

public class Main {
    public static boolean inRange(int[] loc) {
        return 0 <= loc[0] && loc[0] <= 4 && 0 <= loc[1] && loc[1] <= 4;
    }

    public static int solve(int[][] A, int[] aloc, int appleNum) {
        if (appleNum == 0) return 0;
        int ret = -1;
        int[] dr = { -1, 1, 0, 0 };
        int[] dc = { 0, 0, -1, 1 };
        for (int d = 0; d < 4; d++) {
            int[] nloc = { aloc[0] + dr[d], aloc[1] + dc[d] };
            if (inRange(nloc) && A[nloc[0]][nloc[1]] != -1) {
                int prvValue = A[aloc[0]][aloc[1]];
                A[aloc[0]][aloc[1]] = -1;
                int curRet = solve(A, nloc, appleNum - A[nloc[0]][nloc[1]]);
                if (curRet != -1) {
                    curRet++;
                }
                if (curRet != -1) {
                    if (ret == -1 || curRet < ret) {
                        ret = curRet;
                    }
                }
                A[aloc[0]][aloc[1]] = prvValue;
            }
        }
        return ret;
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int[][] A = new int[5][5];
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                A[i][j] = sc.nextInt();
            }
        }
        int[] aloc = new int[2];
        aloc[0] = sc.nextInt();
        aloc[1] = sc.nextInt();
        System.out.println(solve(A, aloc, 3));
    }
}
