import java.util.Scanner;

public class Main {
    static int[] dr = {0, 0, -1, 1};
    static int[] dc = {-1, 1, 0, 0};
    static int[] point = {0, 1, 10, 100, 1000};
    static int N;
    static int[][] A;
    static int[][] B;

    static boolean inRange(int r, int c) {
        return 1 <= r && r <= N && 1 <= c && c <= N;
    }

    static void setStudent(int x, int y1, int y2, int y3, int y4) {
        int r, c, i;
        int mx_r = 1, mx_c = 1, mx_empty = -1, mx_favorate = -1;
        for (r = 1; r <= N; r++) {
            for (c = 1; c <= N; c++) {
                if (A[r][c] != 0) continue;
                int empty = 0, favorate = 0;
                for (i = 0; i < 4; i++) {
                    int nr = r + dr[i], nc = c + dc[i];
                    if (!inRange(nr, nc)) continue;
                    if (A[nr][nc] == 0) empty++;
                    else if (A[nr][nc] == y1 || A[nr][nc] == y2 || A[nr][nc] == y3 || A[nr][nc] == y4) favorate++;
                }
                if (mx_favorate < favorate || (mx_favorate == favorate && mx_empty < empty)) {
                    mx_r = r;
                    mx_c = c;
                    mx_empty = empty;
                    mx_favorate = favorate;
                }
            }
        }
        A[mx_r][mx_c] = x;
    }

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        N = scanner.nextInt();
        A = new int[N + 1][N + 1];
        B = new int[401][4];
        for (int i = 0; i < N * N; i++) {
            int x = scanner.nextInt();
            int y1 = scanner.nextInt();
            int y2 = scanner.nextInt();
            int y3 = scanner.nextInt();
            int y4 = scanner.nextInt();
            B[x][0] = y1;
            B[x][1] = y2;
            B[x][2] = y3;
            B[x][3] = y4;
            setStudent(x, y1, y2, y3, y4);
        }
        int ans = 0;
        for (int r = 1; r <= N; r++) {
            for (int c = 1; c <= N; c++) {
                int student = A[r][c];
                int favorate = 0;
                for (int i = 0; i < 4; i++) {
                    int nr = r + dr[i];
                    int nc = c + dc[i];
                    if (!inRange(nr, nc)) continue;
                    else if (A[nr][nc] == B[student][0] || A[nr][nc] == B[student][1] || A[nr][nc] == B[student][2] || A[nr][nc] == B[student][3]) favorate++;
                }
                ans += point[favorate];
            }
        }
        System.out.println(ans);
    }
}
