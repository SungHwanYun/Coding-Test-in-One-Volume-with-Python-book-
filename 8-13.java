import java.util.*;

class Main {
    public static void add_query(ArrayList<Long> T, int i, int j) {
        T.set(i, T.get(i) + 1);
        T.set(j, T.get(j) - 1);
    }

    public static long get_max_range(ArrayList<Long> R, int range_len) {
        long ret = 0;
        for (int j = range_len - 1; j < 24 * 60 * 60; j++) {
            int i = j - range_len + 1;
            long a = R.get(j);
            if (i != 0) {
                a -= R.get(i - 1);
            }
            ret = Math.max(ret, a);
        }
        return ret;
    }

    public static int translate_time(String t) {
        return Integer.parseInt(t.substring(0, 2)) * 3600 + Integer.parseInt(t.substring(3, 5)) * 60 + Integer.parseInt(t.substring(6));
    }

    public static long solution(int n, ArrayList<ArrayList<String>> A) {
        ArrayList<Long> T = new ArrayList<>(Collections.nCopies(24 * 60 * 60, 0L));
        ArrayList<Long> R = new ArrayList<>(Collections.nCopies(24 * 60 * 60, 0L));
        long answer = 0;
        for (ArrayList<String> a : A) {
            if (a.get(0).equals("1")) {
                add_query(T, translate_time(a.get(1)), translate_time(a.get(2)));
            } else {
                for (int t = 1; t < 24 * 60 * 60; t++) {
                    T.set(t, T.get(t) + T.get(t - 1));
                }
                R.set(0, T.get(0));
                for (int t = 1; t < 24 * 60 * 60; t++) {
                    R.set(t, R.get(t - 1) + T.get(t));
                }
                answer = get_max_range(R, translate_time(a.get(1)));
            }
        }
        return answer;
    }

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int n = scanner.nextInt();
        ArrayList<ArrayList<String>> A = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            ArrayList<String> temp = new ArrayList<>();
            temp.add(scanner.next());
            if (temp.get(0).equals("1")) {
                temp.add(scanner.next());
                temp.add(scanner.next());
            } else {
                temp.add(scanner.next());
            }
            A.add(temp);
        }
        System.out.println(solution(n, A));
    }
}
