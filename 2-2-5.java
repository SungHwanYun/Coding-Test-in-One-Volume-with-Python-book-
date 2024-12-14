import java.util.Scanner;
import java.util.ArrayList;

public class Main {
    public static int parse_log(String s) {
        String hour = s.substring(0, 2);
        String minute = s.substring(3, 5);
        return Integer.parseInt(hour) * 60 + Integer.parseInt(minute);
    }

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        ArrayList<String> A = new ArrayList<String>();
        do {
            String a = scanner.next();
            A.add(a);
        } while (scanner.hasNext());
        int total_time = 0;
        for (String a : A) {
            int t = parse_log(a);
            total_time += t;
        }
        int hour = total_time / 60;
        int minute = total_time % 60;
        String buf;
        if (hour < 100) buf = String.format("%02d:%02d", hour, minute);
        else buf = String.format("%d:%02d", hour, minute);
        System.out.println(buf);
    }
}
