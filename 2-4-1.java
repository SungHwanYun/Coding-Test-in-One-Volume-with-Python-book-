import java.util.*;

public class Main {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        Pair<Integer, Integer> answer = new Pair<>(0, 0);
        int n = scanner.nextInt();
        Deque<Integer> q = new ArrayDeque<>();
        for (int i = 0; i < n; i++) {
            int op = scanner.nextInt();
            if (op == 1) {
                int a = scanner.nextInt();
                q.addLast(a);
                if (answer.getFirst() < q.size() || (answer.getFirst() == q.size() && answer.getSecond() > q.getLast()))
                    answer = new Pair<>(q.size(), q.getLast());
            } else {
                q.removeFirst();
            }
        }
        System.out.println(answer.getFirst() + " " + answer.getSecond());
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