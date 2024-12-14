import java.util.Scanner;
import java.util.ArrayList;

public class Main {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int n = scanner.nextInt();
        int k = scanner.nextInt();
        ArrayList<ArrayList<Integer>> A = new ArrayList<ArrayList<Integer>>(n);
        for (int i = 0; i < n; i++) {
            ArrayList<Integer> row = new ArrayList<Integer>(n);
            for (int j = 0; j < n; j++) {
                row.add(scanner.nextInt());
            }
            A.add(row);
        }
        int answer = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (A.get(i).get(j) == k) {
                    answer++;
                }
            }
        }
        System.out.println(answer);
    }
}
