import java.util.*;

public class Main {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int n = scanner.nextInt();
        List<Integer> A = new ArrayList<>();
        Map<Integer, Integer> D = new HashMap<>();
        for (int i = 0; i < n; i++) {
            int a = scanner.nextInt();
            A.add(a);
            D.put(a, D.getOrDefault(a, 0) + 1);
        }
        int mx = -1;
        for (Map.Entry<Integer, Integer> entry : D.entrySet()) {
            mx = Math.max(mx, entry.getValue());
        }
        List<Integer> answer = new ArrayList<>();
        for (Map.Entry<Integer, Integer> entry : D.entrySet()) {
            if (entry.getValue() == mx) {
                answer.add(entry.getKey());
            }
        }
        Collections.sort(answer);
        for (int i = 0; i < answer.size(); i++) {
            System.out.print(answer.get(i));
            if (i < answer.size() - 1) {
                System.out.print(" ");
            }
        }
    }
}
