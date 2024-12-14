import java.util.*;

public class Main {
    public static List<String> split(String s, char delimiter) {
        List<String> tokens = new ArrayList<>();
        StringBuilder token = new StringBuilder();
        for (char c : s.toCharArray()) {
            if (c == delimiter) {
                tokens.add(token.toString());
                token = new StringBuilder();
            } else {
                token.append(c);
            }
        }
        tokens.add(token.toString());
        return tokens;
    }

    public static Pair<Integer, String> parse_log(String s) {
        List<String> tokens = split(s, ' ');
        int t = Integer.parseInt(tokens.get(0).substring(0, 2)) * 60 + Integer.parseInt(tokens.get(0).substring(3, 5));
        String name = tokens.get(1);
        return new Pair<>(t, name);
    }

    public static int get_fee(List<Integer> fees, int t) {
        int money = fees.get(1);
        if (fees.get(0) < t) {
            money += (t - fees.get(0) + fees.get(2) - 1) / fees.get(2) * fees.get(3);
        }
        return money;
    }

    public static List<Pair<String, Integer>> solution(int n, List<String> A, List<Integer> fees) {
        Map<String, Integer> d = new HashMap<>();
        for (String log : A) {
            Pair<Integer, String> parsed_log = parse_log(log);
            int t = parsed_log.getFirst();
            String name = parsed_log.getSecond();
            if (d.containsKey(name)) {
                d.put(name, d.get(name) + t);
            } else {
                d.put(name, t);
            }
        }
        List<Pair<String, Integer>> answer = new ArrayList<>();
        for (Map.Entry<String, Integer> entry : d.entrySet()) {
            entry.setValue(get_fee(fees, entry.getValue()));
            answer.add(new Pair(entry.getKey(), entry.getValue()));
        }
        
        answer.sort((a, b) -> {
            if (!a.getSecond().equals(b.getSecond())) {
                return b.getSecond() - a.getSecond();
            } else {
                return a.getFirst().compareTo(b.getFirst());
            }
        });
        return answer;
    }

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int n = scanner.nextInt();
        scanner.nextLine();
        List<String> A = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            A.add(scanner.nextLine());
        }
        List<Integer> fees = Arrays.asList(100, 10, 50, 3);
        List<Pair<String, Integer>> B = solution(n, A, fees);
        for (Pair<String, Integer> entry : B) {
            System.out.println(entry.getFirst() + " " + entry.getSecond());
        }
    }
}

class Pair<T, U> {
    private T first;
    private U second;

    public Pair(T first, U second) {
        this.first = first;
        this.second = second;
    }

    public T getFirst() {
        return first;
    }

    public U getSecond() {
        return second;
    }
}