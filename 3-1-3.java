import java.util.Scanner;

public class Main {
    public static boolean inRange(int r, int c) {
        return 0 <= r && r <= 4 && 0 <= c && c <= 4;
    }

    public static void main(String[] args) {
        int[][] A = new int[5][5];
        Scanner scanner = new Scanner(System.in);
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                A[i][j] = scanner.nextInt();
            }
        }
        int r = scanner.nextInt();
        int c = scanner.nextInt();
        int[] dr = { -1, 1, 0, 0 };
        int[] dc = { 0, 0, -1, 1 };
        for (int i = 0; i < 4; i++) {
            int nr = r + dr[i];
            int nc = c + dc[i];
            if (inRange(nr, nc) && A[nr][nc] == 1) {
                System.out.println(1);
                System.exit(0);
            }
        }
        System.out.println(0);
    }
}
