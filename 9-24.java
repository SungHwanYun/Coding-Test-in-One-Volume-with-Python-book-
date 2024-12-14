import java.util.*;

enum DIR {
    LEFT(1), LEFT_UP(2), UP(3), RIGHT_UP(4), RIGHT(5), RIGHT_DOWN(6), DOWN(7), LEFT_DOWN(8);
    private final int value;
    
    private DIR(int value) {
        this.value = value;
    }
    
    public int getValue() {
        return value;
    }
}

public class Main {
    static int[][][] A = new int[5][5][10];
    static int[][][] C = new int[5][5][104];
    static int[][][] X = new int[5][5][10];
    static int[] dr = {0, 0, -1, -1, -1, 0, 1, 1, 1};
    static int[] dc = {0, -1, -1, 0, 1, 1, 1, 0, -1};
    static int N = 4, Sr, Sc;
    
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int m = sc.nextInt();
        int s = sc.nextInt();
        
        while (m-- > 0) {
            int r = sc.nextInt();
            int c = sc.nextInt();
            int d = sc.nextInt();
            A[r][c][d]++;
        }
        
        Sr = sc.nextInt();
        Sc = sc.nextInt();
        
        for (int r = 1; r <= N; r++) {
            for (int c = 1; c <= N; c++) {
                for (int d = 1; d <= 8; d++) {
                    A[r][c][0] += A[r][c][d];
                }
            }
        }
        
        for (int step = 1; step <= s; step++) {
            copy_A();
            move_fish(step);
            move_shark(step);
            paste_A();
        }
        
        int ans = 0;
        for (int r = 1; r <= N; r++) {
            for (int c = 1; c <= N; c++) {
                ans += A[r][c][0];
            }
        }
        
        System.out.println(ans);
    }
    
    static boolean in_range(int r, int c) {
        return 1 <= r && r <= N && 1 <= c && c <= N;
    }
    
    static void copy_A() {
        for (int r = 1; r <= N; r++) {
            for (int c = 1; c <= N; c++) {
                for (int d = 0; d <= 8; d++) {
                    X[r][c][d] = A[r][c][d];
                }
            }
        }
    }
    
    static void paste_A() {
        for (int r = 1; r <= N; r++) {
            for (int c = 1; c <= N; c++) {
                for (int d = 0; d <= 8; d++) {
                    A[r][c][d] += X[r][c][d];
                }
            }
        }
    }
    
    static int is_smell(int step, int r, int c) {
        if (step == 1) {
            return 0;
        }
        if (step == 2) {
            return C[r][c][1];
        }
        return C[r][c][step - 1] | C[r][c][step - 2];
    }
    
    static void move_fish(int step) {
        int[][][] T = new int[5][5][10];
        
        for (int r = 1; r <= N; r++) {
            for (int c = 1; c <= N; c++) {
                for (int d = 0; d <= 8; d++) {
                    T[r][c][d] = 0;
                }
            }
        }
        
        for (int r = 1; r <= N; r++) {
            for (int c = 1; c <= N; c++) {
                for (int d = 1; d <= 8; d++) {
                    if (A[r][c][d] == 0) {
                        continue;
                    }
                    
                    int nr = 0, nc = 0, nd = 0;
                    int i;
                    for (i = 0; i < 8; i++) {
                        nd = d - i;
                        if (nd <= 0) {
                            nd += 8;
                        }
                        nr = r + dr[nd];
                        nc = c + dc[nd];
                        if (in_range(nr, nc) && !(nr == Sr && nc == Sc) && is_smell(step, nr, nc) == 0) {
                            break;
                        }
                    }
                    
                    if (i == 8) {
                        nr = r;
                        nc = c;
                        nd = d;
                    }
                    
                    T[nr][nc][nd] += A[r][c][d];
                    T[nr][nc][0] += A[r][c][d];
                }
            }
        }
        
        for (int r = 1; r <= N; r++) {
            for (int c = 1; c <= N; c++) {
                for (int d = 0; d <= 8; d++) {
                    A[r][c][d] = T[r][c][d];
                }
            }
        }
    }
    
    static void reset_A(int r, int c) {
        for (int d = 0; d <= 8; d++) {
            A[r][c][d] = 0;
        }
    }
    
    static int[] rr = {-1, 0, 1, 0};
    static int[] cc = {0, -1, 0, 1};
    
    static int get_sum(int d1, int d2, int d3) {
        int[][] visited = new int[5][5];
        visited[Sr + rr[d1]][Sc + cc[d1]] = 1;
        visited[Sr + rr[d1] + rr[d2]][Sc + cc[d1] + cc[d2]] = 1;
        visited[Sr + rr[d1] + rr[d2] + rr[d3]][Sc + cc[d1] + cc[d2] + cc[d3]] = 1;
        int ret = 0;
        for (int r = 1; r <= N; r++) {
            for (int c = 1; c <= N; c++) {
                if (visited[r][c] != 0) {
                    ret += A[r][c][0];
                }
            }
        }
        return ret;
    }
    
    static void move_shark(int step) {
        int x = 0, y = 0, z = 0, mx = -1, r, c;
        
        for (int i = 0; i < 4; i++) {
            r = Sr + rr[i];
            c = Sc + cc[i];
            if (!in_range(r, c)) {
                continue;
            }
            
            for (int j = 0; j < 4; j++) {
                r = Sr + rr[i] + rr[j];
                c = Sc + cc[i] + cc[j];
                if (!in_range(r, c)) {
                    continue;
                }
                
                for (int k = 0; k < 4; k++) {
                    r = Sr + rr[i] + rr[j] + rr[k];
                    c = Sc + cc[i] + cc[j] + cc[k];
                    if (!in_range(r, c)) {
                        continue;
                    }
                    
                    int sum = get_sum(i, j, k);
                    if (sum > mx) {
                        mx = sum;
                        x = i;
                        y = j;
                        z = k;
                    }
                }
            }
        }
        
        r = Sr + rr[x];
        c = Sc + cc[x];
        if (A[r][c][0] > 0) {
            reset_A(r, c);
            C[r][c][step] = 1;
        }
        
        r += rr[y];
        c += cc[y];
        if (A[r][c][0] > 0) {
            reset_A(r, c);
            C[r][c][step] = 1;
        }
        
        r += rr[z];
        c += cc[z];
        if (A[r][c][0] > 0) {
            reset_A(r, c);
            C[r][c][step] = 1;
        }
        
        Sr = r;
        Sc = c;
    }
}
