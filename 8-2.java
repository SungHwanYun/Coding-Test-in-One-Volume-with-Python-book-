import java.util.*;

public class Main {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int n = scanner.nextInt();
        scanner.nextLine();
        String A = scanner.nextLine();
        List<String> B = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            B.add(scanner.nextLine());
        }
        Map<String, Integer> d = new HashMap<>();
        String temp = "";
        for (int i = 0; i < A.length(); i++) {
            if (A.charAt(i) == ' ') {
                d.put(temp, 0);
                temp = "";
            } else {
                temp += A.charAt(i);
            }
        }
        d.put(temp, 0);
        for (int i = 0; i < n; i++) {
            temp = "";
            for (int j = 0; j < B.get(i).length(); j++) {
                if (B.get(i).charAt(j) == ' ') {
                    d.put(temp, d.getOrDefault(temp, 0) + 1);
                    temp = "";
                } else {
                    temp += B.get(i).charAt(j);
                }
            }
            d.put(temp, d.getOrDefault(temp, 0) + 1);
        }
        List<Map.Entry<String, Integer>> answer = new ArrayList<>(d.entrySet());
        Collections.sort(answer, (a, b) -> {
            if (!a.getValue().equals(b.getValue())) {
                return b.getValue() - a.getValue();
            } else {
                return a.getKey().compareTo(b.getKey());
            }
        });
        for (Map.Entry<String, Integer> entry : answer) {
            System.out.println(entry.getKey() + " " + entry.getValue());
        }
    }
}
