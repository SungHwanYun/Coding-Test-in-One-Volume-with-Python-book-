import java.util.*;

public class Main {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        List<Integer>[] answer = new ArrayList[3];
        for (int i = 0; i < 3; i++) {
            answer[i] = new ArrayList<>();
        }
        int n = sc.nextInt();
        Deque<Pair<Integer, Integer>> q = new ArrayDeque<>();
        for (int i = 0; i < n; i++) {
            int op = sc.nextInt();
            if (op == 1) {
                int a = sc.nextInt();
                int b = sc.nextInt();
                q.addLast(new Pair<>(a, b));
            } else {
                int a = q.getFirst().getFirst();
                int b = q.getFirst().getSecond();
                q.removeFirst();
                int bb = sc.nextInt();
                if (b == bb) {
                    answer[0].add(a);
                } else {
                    answer[1].add(a);
                }
            }
        }
        while (!q.isEmpty()) {
            int a = q.getFirst().getFirst();
            q.removeFirst();
            answer[2].add(a);
        }
        for (int i = 0; i < 3; i++) {
            Collections.sort(answer[i]);
            if (answer[i].size() == 0) {
                System.out.println("None");
            } else {
                for (int j = 0; j < answer[i].size(); j++) {
                    System.out.print(answer[i].get(j));
                    if (j < answer[i].size() - 1) {
                        System.out.print(" ");
                    } else {
                        System.out.println();
                    }
                }
            }
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