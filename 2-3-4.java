import java.util.Scanner;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class Main {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        ArrayList<String> A = new ArrayList<>();
        
        Scanner lineScanner = new Scanner(scanner.nextLine());
        while (lineScanner.hasNext()) {
            String s = lineScanner.next();
            A.add(s);
        }
        lineScanner.close();

        String B = scanner.next();
        Map<String, Integer> D = new HashMap<>();
        for (String a : A) {
            for (int i = 1; i < a.length(); i++) {
                String s = a.substring(0, i);
                D.put(s, D.getOrDefault(s, 0) + 1);
            }
        }
        System.out.println(D.getOrDefault(B, 0));
    }
}