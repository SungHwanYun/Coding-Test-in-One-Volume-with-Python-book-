import java.util.Scanner;

public class Main {
    public static boolean inRange(int[] loc) {
        return 0 <= loc[0] && loc[0] <= 4 && 0 <= loc[1] && loc[1] <= 4;
    }

    public static int getApple(int[][] A, int[] aloc, int[] iloc, int[] jloc, int[] kloc) {
        int appleNum = 0;
        if (!inRange(iloc) || !inRange(jloc)) return 0;
        if (A[iloc[0]][iloc[1]] == -1 || A[jloc[0]][jloc[1]] == -1) return 0;
        if (aloc[0] == jloc[0] && aloc[1] == jloc[1]) return 0;
        appleNum = A[iloc[0]][iloc[1]] + A[jloc[0]][jloc[1]];
        if (inRange(kloc) && A[kloc[0]][kloc[1]] == 1 &&
                (iloc[0] != kloc[0] || iloc[1] != kloc[1]))
            appleNum += 1;
        return appleNum;
    }

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int n = 5;
        int[][] A = new int[5][5];
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                A[i][j] = scanner.nextInt();
            }
        }
        int[] aloc = new int[2];
        aloc[0] = scanner.nextInt();
        aloc[1] = scanner.nextInt();
        int[] dr = {-1, 1, 0, 0};
        int[] dc = {0, 0, -1, 1};
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                for (int k = 0; k < 4; k++) {
                    int[] iloc = {aloc[0] + dr[i], aloc[1] + dc[i]};
                    int[] jloc = {iloc[0] + dr[j], iloc[1] + dc[j]};
                    int[] kloc = {jloc[0] + dr[k], jloc[1] + dc[k]};
                    if (getApple(A, aloc, iloc, jloc, kloc) >= 2) {
                        System.out.println(1);
                        System.exit(0);
                    }
                }
            }
        }
        System.out.println(0);
    }
}
