import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.Collections;

public class Main {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        List<String> A = new ArrayList<>();
        Scanner lineScanner = new Scanner(scanner.nextLine());
        while (lineScanner.hasNext()) {
            String s = lineScanner.next();
            A.add(s);
        }
        lineScanner.close();
        
        lineScanner = new Scanner(scanner.nextLine());
        List<String> B = new ArrayList<>();
        while (lineScanner.hasNext()) {
            String s = lineScanner.next();
            B.add(s);
        }
        lineScanner.close();

        Map<String, Integer> D = new HashMap<>();
        for (String b : B) {
            D.put(b, D.getOrDefault(b, 0) + 1);
        }
        List<String> answer = new ArrayList<>();
        for (String a : A) {
            if (D.getOrDefault(a, 0) == 0) {
                answer.add(a);
            }
        }
        Collections.sort(answer);
        for (String a : answer) {
            System.out.println(a);
        }
    }
}