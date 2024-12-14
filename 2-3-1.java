import java.util.*;

public class Main {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int n = scanner.nextInt();
        int m = scanner.nextInt();
        Map<String, Integer> D = new HashMap<>();
        for (int i = 0; i < n; i++) {
            String s = scanner.next();
            int cost = scanner.nextInt();
            D.put(s, cost);
        }
        List<String> B = new ArrayList<>();
        for (int i = 0; i < m; i++) {
            String s = scanner.next();
            B.add(s);
        }
        
        long answer = 0;
        for (String b : B) {
            answer += D.get(b);
        }
        System.out.println(answer);
    }
}