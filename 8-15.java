import java.util.*;

public class Main {
    public static boolean inRange(int r, int c) {
        return 0 <= r && r <= 4 && 0 <= c && c <= 4;
    }

    public static int getMoveCount(List<List<Integer>> board, int sr, int sc, int tr, int tc) {
        int[][] dd = { {0, -1}, {0, 1}, {-1, 0}, {1, 0} };
        int[][] visited = new int[5][5];
        int[][] dist = new int[5][5];
        Queue<int[]> q = new LinkedList<>();
        q.add(new int[]{ sr, sc });
        visited[sr][sc] = 1;
        while (!q.isEmpty()) {
            int[] front = q.poll();
            int r = front[0], c = front[1];
            if (r == tr && c == tc) {
                return dist[r][c];
            }
            for (int[] d : dd) {
                int nr = r + d[0], nc = c + d[1];
                if (inRange(nr, nc) && visited[nr][nc] == 0 && board.get(nr).get(nc) != -1) {
                    q.add(new int[]{ nr, nc });
                    dist[nr][nc] = dist[r][c] + 1;
                    visited[nr][nc] = 1;
                }
            }
            for (int[] d : dd) {
                int nr = r, nc = c;
                while (true) {
                    if (!inRange(nr + d[0], nc + d[1])) {
                        break;
                    }
                    if (board.get(nr + d[0]).get(nc + d[1]) == -1) {
                        break;
                    }
                    nr += d[0];
                    nc += d[1];
                    if (board.get(nr).get(nc) == 7) {
                        break;
                    }
                }
                if (visited[nr][nc] == 0) {
                    q.add(new int[]{ nr, nc });
                    dist[nr][nc] = dist[r][c] + 1;
                    visited[nr][nc] = 1;
                }
            }
        }
        return -1;
    }

    public static int solution(List<List<Integer>> board, int sr, int sc) {
        List<List<Integer>> source = new ArrayList<>();
        for (int i = 0; i < 6; i++) {
            source.add(new ArrayList<>());
        }
        for (int r = 0; r < 5; r++) {
            for (int c = 0; c < 5; c++) {
                if (board.get(r).get(c) > 0 && board.get(r).get(c) < 7) {
                    source.get(board.get(r).get(c) - 1).add(r);
                    source.get(board.get(r).get(c) - 1).add(c);
                }
            }
        }
        source.sort(Comparator.comparingInt(List::size));

        int[] v = new int[6];
        for (int i=0; i<6; i++) v[i]=i;

        int answer = -1;
        int cnt = 0;
        do {
            int ret = 0;
            int r = sr, c = sc;
            for (int i = 0; i < v.length; i++) {
                int nr = source.get(v[i]).get(0), nc = source.get(v[i]).get(1);
                int x = getMoveCount(board, r, c, nr, nc);
                if (x == -1) {
                    ret = -1;
                    break;
                }
                ret += x;
                r = nr;
                c = nc;
            }
            if (ret != -1) {
                if (answer == -1 || answer > ret) {
                    answer = ret;
                }
            }
        } while (next_permutation(v));
        return answer;
    }

    public static boolean next_permutation(int[] v) {
        int n = v.length;
        int i = n - 1;
        while (i > 0 && v[i - 1] >= v[i]) i--;
        if (i <= 0) return false;
        int j = n - 1;
        while (v[j] <= v[i - 1]) j--;
        int temp = v[i-1];
        v[i-1]=v[j];
        v[j]=temp;
        j = n - 1;
        while (i < j) {
            temp = v[i];
            v[i] = v[j];
            v[j] = temp;
            i++;
            j--;
        }
        return true;
    }

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        List<List<Integer>> board = new ArrayList<>();
        for (int i = 0; i < 5; i++) {
            board.add(new ArrayList<>());
            for (int j = 0; j < 5; j++) {
                board.get(i).add(scanner.nextInt());
            }
        }
        int sr = scanner.nextInt();
        int sc = scanner.nextInt();
        System.out.println(solution(board, sr, sc));
    }
}
