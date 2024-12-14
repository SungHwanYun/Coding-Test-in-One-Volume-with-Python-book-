import java.util.*;

public class Main {
    public static boolean inRange(int r, int c) {
        return 0 <= r && r <= 4 && 0 <= c && c <= 4;
    }

    public static int getMoveCount(List<List<Integer>> board, int sr, int sc, int tr, int tc) {
        List<List<Integer>> dd = Arrays.asList(Arrays.asList(0, -1), Arrays.asList(0, 1), Arrays.asList(-1, 0), Arrays.asList(1, 0));
        List<List<Integer>> visited = new ArrayList<>();
        List<List<Integer>> dist = new ArrayList<>();
        for (int i = 0; i < 5; i++) {
            visited.add(new ArrayList<>(Arrays.asList(0, 0, 0, 0, 0)));
            dist.add(new ArrayList<>(Arrays.asList(0, 0, 0, 0, 0)));
        }
        Deque<List<Integer>> q = new ArrayDeque<>();
        q.add(Arrays.asList(sr, sc));
        visited.get(sr).set(sc, 1);
        while (!q.isEmpty()) {
            int r = q.getFirst().get(0), c = q.getFirst().get(1);
            q.removeFirst();
            if (r == tr && c == tc) {
                return dist.get(r).get(c);
            }
            for (List<Integer> d : dd) {
                int nr = r + d.get(0), nc = c + d.get(1);
                if (inRange(nr, nc) && visited.get(nr).get(nc) == 0 && board.get(nr).get(nc) != -1) {
                    q.add(Arrays.asList(nr, nc));
                    dist.get(nr).set(nc, dist.get(r).get(c) + 1);
                    visited.get(nr).set(nc, 1);
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
                if (board.get(r).get(c) > 0) {
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
