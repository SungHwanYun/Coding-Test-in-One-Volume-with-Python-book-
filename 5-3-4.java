import java.util.Scanner;

public class Main {
    public static int translateTime(String t) {
        int x = Integer.parseInt(t.substring(0, 2)) * 60 + Integer.parseInt(t.substring(3, 5));
        assert (0 <= x && x < 3600);
        return x;
    }

    public static int getFee(int[] fees, int t) {
        int money = fees[1];
        if (fees[0] < t) {
            money += (t - fees[0] + fees[2] - 1) / fees[2] * fees[3];
        }
        return money;
    }

    public static void main(String[] args) {
        int[] fees = {100, 10, 50, 3};
        int totalCost = 0;
        Scanner scanner = new Scanner(System.in);
        do {
            String t = scanner.next();
            totalCost += getFee(fees, translateTime(t));
        } while (scanner.hasNext());
        System.out.println(totalCost);
    }
}
