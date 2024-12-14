import java.util.*;

public class Main {
    public static int solution(ArrayList<ArrayList<Integer>> A) {
        ArrayList<ArrayList<Integer>> D = new ArrayList<>();
        for (int i = 0; i < 6; i++) {
            ArrayList<Integer> row = new ArrayList<>(Arrays.asList(0, 0));
            D.add(row);
        }
        D.get(0).set(0, A.get(0).get(1));
        D.get(0).set(1, A.get(0).get(1));
        for (int i = 1; i < 6; i++) {
            D.get(i).set(0, Math.min(D.get(i - 1).get(0) + A.get(2 * i - 2).get(2 * i + 1) + A.get(2 * i).get(2 * i + 1),
                    D.get(i - 1).get(1) + A.get(2 * i - 1).get(2 * i + 1) + A.get(2 * i).get(2 * i + 1)));
            D.get(i).set(1, Math.min(D.get(i - 1).get(0) + A.get(2 * i - 2).get(2 * i) + A.get(2 * i).get(2 * i + 1),
                    D.get(i - 1).get(1) + A.get(2 * i - 1).get(2 * i) + A.get(2 * i).get(2 * i + 1)));
        }
        return Math.min(D.get(5).get(0), D.get(5).get(1));
    }

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        ArrayList<ArrayList<Integer>> A = new ArrayList<>();
        for (int i = 0; i < 12; i++) {
            ArrayList<Integer> row = new ArrayList<>();
            for (int j = 0; j < 12; j++) {
                row.add(scanner.nextInt());
            }
            A.add(row);
        }
        System.out.println(solution(A));
    }
}
