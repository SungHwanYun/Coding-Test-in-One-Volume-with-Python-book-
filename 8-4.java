import java.util.*;

public class Main {
    public static void solve(List<String> C, Map<String, Integer> d) {
        for (String c : C) {
            if (d.containsKey(c)) {
                d.put(c, d.get(c) + 1);
            } else {
                d.put(c, 1);
            }
        }
    }

    public static List<String> get_combinations(String s, int r) {
        int n = s.length();
        int[] v = new int[n];
        Arrays.fill(v, 0, r, 1);
        List<String> ans = new ArrayList<>();
        do {
            StringBuilder x = new StringBuilder();
            for (int i = 0; i < n; ++i) {
                if (v[i] == 1) {
                    x.append(s.charAt(i));
                }
            }
            ans.add(x.toString());
        } while (prev_permutation(v));
        return ans;
    }

    public static boolean prev_permutation(int[] v) {
        int n = v.length;
        int i = n - 1;
        while (i > 0 && v[i - 1] <= v[i]) {
            i--;
        }
        if (i <= 0) {
            return false;
        }
        int j = n - 1;
        while (v[j] >= v[i - 1]) {
            j--;
        }
        int temp = v[i - 1];
        v[i - 1] = v[j];
        v[j] = temp;
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

    public static List<String> solution(String X, String Y, String Z, int k) {
        List<String> CX, CY, CZ;
        CX = get_combinations(X, k);
        CY = get_combinations(Y, k);
        CZ = get_combinations(Z, k);
        Map<String, Integer> d = new HashMap<>();
        solve(CX, d);
        solve(CY, d);
        solve(CZ, d);
        List<String> answer = new ArrayList<>();
        for (Map.Entry<String, Integer> entry : d.entrySet()) {
            if (entry.getValue() >= 2) {
                answer.add(entry.getKey());
            }
        }
        Collections.sort(answer);
        if (answer.size() == 0) {
            answer.add("-1");
        }
        return answer;
    }

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        String X = scanner.next();
        String Y = scanner.next();
        String Z = scanner.next();
        int k = scanner.nextInt();
        List<String> C = solution(X, Y, Z, k);
        for (String c : C) {
            System.out.println(c);
        }
    }
}