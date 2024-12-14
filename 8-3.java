import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class Main {
    public static boolean is_ok(List<String> qry, List<String> student) {
        for (int i = 0; i < 3; i++) {
            if (!qry.get(i).equals("-") && !qry.get(i).equals(student.get(i))) {
                return false;
            }
        }
        return true;
    }

    public static List<Integer> solution(int n, int m, List<List<String>> A, List<List<String>> B) {
        List<Integer> answer = new ArrayList<>();
        for (List<String> qry : B) {
            int cnt = 0;
            for (List<String> student : A) {
                if (is_ok(qry, student)) {
                    cnt += 1;
                }
            }
            answer.add(cnt);
        }
        return answer;
    }

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int n = scanner.nextInt();
        int m = scanner.nextInt();
        List<List<String>> A = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            List<String> row = new ArrayList<>();
            for (int j = 0; j < 3; j++) {
                row.add(scanner.next());
            }
            A.add(row);
        }
        List<List<String>> B = new ArrayList<>();
        for (int i = 0; i < m; i++) {
            List<String> row = new ArrayList<>();
            for (int j = 0; j < 3; j++) {
                row.add(scanner.next());
            }
            B.add(row);
        }
        List<Integer> C = solution(n, m, A, B);
        for (int c : C) {
            System.out.println(c);
        }
    }
}
