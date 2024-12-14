import java.util.*;

class Main {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        List<List<Integer>> board = new ArrayList<>();
        for (int i = 0; i < 5; i++) {
            List<Integer> row = new ArrayList<>();
            for (int j = 0; j < 5; j++) {
                int num = scanner.nextInt();
                row.add(num);
            }
            board.add(row);
        }
        List<Integer> loc = new ArrayList<>();
        for (int i = 0; i < 4; i++) {
            int num = scanner.nextInt();
            loc.add(num);
        }
        List<Integer> aloc = loc.subList(0, 2);
        List<Integer> bloc = loc.subList(2, 4);
        System.out.println(solution(board, aloc, bloc));
    }

    public static int solution(List<List<Integer>> board, List<Integer> aloc, List<Integer> bloc) {
        return solve(board, aloc, bloc, 0);
    }

    public static int solve(List<List<Integer>> board, List<Integer> aloc, List<Integer> bloc, int apple_diff) {
        if (board.get(aloc.get(0)).get(aloc.get(1)) == -1 && board.get(bloc.get(0)).get(bloc.get(1)) == -1) {
            if (apple_diff > 0) {
                return 1;
            }
            return 0;
        }
        int remained_apple = 0;
        for (int i = 0; i < 5; i++) {
            remained_apple += Collections.frequency(board.get(i), 1);
        }
        if (remained_apple == 0) {
            if (apple_diff > 0) {
                return 1;
            }
            return 0;
        }
        List<List<Integer>> dd = Arrays.asList(Arrays.asList(-1, 0), Arrays.asList(1, 0), Arrays.asList(0, -1), Arrays.asList(0, 1));
        int try_count = 0;
        for (List<Integer> d : dd) {
            int dr = d.get(0);
            int dc = d.get(1);
            int r = aloc.get(0) + dr;
            int c = aloc.get(1) + dc;
            if (in_range(Arrays.asList(r, c)) && board.get(r).get(c) != -1 && !Arrays.asList(r, c).equals(bloc)) {
                try_count += 1;
                int prv_value = board.get(aloc.get(0)).get(aloc.get(1));
                board.get(aloc.get(0)).set(aloc.get(1), -1);
                int ret = solve(board, bloc, Arrays.asList(r, c), -(apple_diff + board.get(r).get(c)) + 1);
                board.get(aloc.get(0)).set(aloc.get(1), prv_value);
                if (ret == 0) {
                    return 1;
                }
            }
        }
        if (try_count == 0) {
            int prv_value = board.get(aloc.get(0)).get(aloc.get(1));
            board.get(aloc.get(0)).set(aloc.get(1), -1);
            int ret = solve(board, bloc, aloc, -apple_diff + 1);
            board.get(aloc.get(0)).set(aloc.get(1), prv_value);
            if (ret == 0) {
                return 1;
            }
        }
        return 0;
    }

    public static boolean in_range(List<Integer> loc) {
        return 0 <= loc.get(0) && loc.get(0) <= 4 && 0 <= loc.get(1) && loc.get(1) <= 4;
    }
}
