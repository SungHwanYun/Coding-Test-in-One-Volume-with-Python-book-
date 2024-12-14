import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        List<Integer> X = new ArrayList<>();
        Scanner scanner = new Scanner(System.in);
        
        do {
            int x = scanner.nextInt();
            X.add(x);
        } while (scanner.hasNextInt());
        
        List<Integer> A = new ArrayList<>();
        List<Integer> B = new ArrayList<>();
        for (int i = 0; i < X.size(); i++) {
            if (i < X.size() / 2) A.add(X.get(i));
            else B.add(X.get(i));
        }
        
        int a = 0, b = 0;
        for (int i = 0; i < A.size(); i++) {
            if (A.get(i) > B.get(i)) {
                a++;
            } else if (A.get(i) < B.get(i)) {
                b++;
            }
        }
        
        if (a > b) {
            System.out.println(1);
        } else {
            System.out.println(0);
        }
    }
}