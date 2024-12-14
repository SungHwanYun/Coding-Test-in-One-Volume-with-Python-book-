import java.util.*;

public class Main {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();
        int[][] D = new int[n + 1][10];
        for (int i = 0; i <= n; i++) {
            Arrays.fill(D[i], 0);
        }
        for (int j = 1; j <= 9; j++) {
            D[1][j] = 1;
        }
        for (int i = 2; i <= n; i++) {
            for (int j = 1; j <= 9; j++) {
                int s = Math.max(j - 2, 1);
                int e = Math.min(j + 2, 9);
                for (int k = s; k <= e; k++) {
                    D[i][j] += D[i - 1][k];
                    D[i][j] %= 987654321;
                }
            }
        }
        int answer = 0;
        for (int j = 1; j <= 9; j++) {
            answer += D[n][j];
            answer %= 987654321;
        }
        System.out.println(answer);
    }
}
