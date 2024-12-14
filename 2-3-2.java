import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        ArrayList<String> S = new ArrayList<>();
        do {
            String s = scanner.next();
            S.add(s);
        } while (scanner.hasNext());
        Map<String, Integer> D = new HashMap<>();
        for (String s : S) {
            D.put(s, D.getOrDefault(s, 0) + 1);
        }
        ArrayList<Map.Entry<String, Integer>> V = new ArrayList<>(D.entrySet());
        V.sort(Map.Entry.comparingByKey());
        for (Map.Entry<String, Integer> v : V) {
            System.out.println(v.getKey() + " " + v.getValue());
        }
    }
}
