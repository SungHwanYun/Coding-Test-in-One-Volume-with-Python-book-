import java.util.*;

class Solution {
    static class Pair {
        int first;
        int second;
        
        public Pair(int first, int second) {
            this.first = first;
            this.second = second;
        }
    }
    
    static int[] dr = {0, 0, -1, 1};
    static int[] dc = {-1, 1, 0, 0};
    static Pair[] A = new Pair[10];
    static Pair[] B = new Pair[10];
    static List<Integer> X = new ArrayList<>();
    static List<Integer> Y = new ArrayList<>();
    static int N, sr, sc;
    static int[] S = new int[10];
    static int answer = (int)1e8;
    
    static boolean inRange(int r, int c) {
        return 0 <= r && r < 4 && 0 <= c && c < 4;
    }
    
    static int bfs(int[][] board, int r1, int c1, int r2, int c2) {
        int[][] visited = new int[4][4];
        int[][] dist = new int[4][4];
        for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) visited[i][j] = 0;
        Queue<Pair> Q = new LinkedList<>();
        Q.add(new Pair(r1, c1));
        dist[r1][c1] = 0;
        visited[r1][c1] = 1;
        while (!Q.isEmpty()) {
            int r = Q.peek().first;
            int c = Q.peek().second;
            Q.poll();
            if (r == r2 && c == c2) {
                return dist[r][c] + 1;
            }
            for (int i = 0; i < 4; i++) {
                int nr = r + dr[i];
                int nc = c + dc[i];
                if (inRange(nr, nc) && visited[nr][nc] == 0) {
                    Q.add(new Pair(nr, nc));
                    dist[nr][nc] = dist[r][c] + 1;
                    visited[nr][nc] = 1;
                }
            }
            for (int i = 0; i < 4; i++) {
                int nr = r;
                int nc = c;
                do {
                    if (inRange(nr + dr[i], nc + dc[i]) == false)
                        break;
                    nr += dr[i];
                    nc += dc[i];
                    if (board[nr][nc] != 0) break;
                } while (true);
                if (visited[nr][nc] == 0) {
                    Q.add(new Pair(nr, nc));
                    dist[nr][nc] = dist[r][c] + 1;
                    visited[nr][nc] = 1;
                }
            }
        }
        return (int)1e8;
    }
    
    static int getMoveCount(int[][] board) {
        int sum = 0;
        int r = sr;
        int c = sc;
        for (int i = 0; i < Y.size(); ++i) {
            int x = Y.get(i);
            int nr1, nc1, nr2, nc2;
            if (x < 10) {
                x = X.get(x);
                nr1 = A[x].first;
                nc1 = A[x].second;
                nr2 = B[x].first;
                nc2 = B[x].second;
            } else {
                x = X.get(x - 10);
                nr1 = B[x].first;
                nc1 = B[x].second;
                nr2 = A[x].first;
                nc2 = A[x].second;
            }
            sum += bfs(board, r, c, nr1, nc1) + bfs(board, nr1, nc1, nr2, nc2);
            r = nr2;
            c = nc2;
            board[nr1][nc1] = 0;
            board[nr2][nc2] = 0;
        }
        return sum;
    }
    
    static void solve(int[][] board) {
        if (Y.size() == N) {
            answer = Math.min(answer, getMoveCount(board));
            return;
        }
        int[][] board_temp = new int[4][4];
        for (int i = 0; i < N; i++) {
            if (S[i] == 1) continue;
            S[i] = 1;
            Y.add(i);
            for (int r = 0; r < 4; r++) for (int c = 0; c < 4; c++) board_temp[r][c] = board[r][c];
            solve(board_temp);
            Y.remove(Y.size() - 1);
            Y.add(10 + i);
            for (int r = 0; r < 4; r++) for (int c = 0; c < 4; c++) board_temp[r][c] = board[r][c];
            solve(board_temp);
            Y.remove(Y.size() - 1);
            S[i] = 0;
        }
    }
    
    public int solution(int[][] board, int r, int c) {
        sr = r;
        sc = c;
        int[] cnt = new int[10];
        for (int i = 0; i < 10; i++) cnt[i] = 0;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                int num = board[i][j];
                if (num != 0) {
                    cnt[num]++;
                    if (cnt[num] == 1) {
                        A[num] = new Pair(i, j);
                    } else {
                        B[num] = new Pair(i, j);
                        X.add(num);
                    }
                }
            }
        }
        Collections.sort(X);
        N = X.size();
        solve(board);
        return answer;
    }
}
