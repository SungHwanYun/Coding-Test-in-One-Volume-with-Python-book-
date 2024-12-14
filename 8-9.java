import java.util.*;

public class Main {
    public static String solution(int n, int k) {
        String a = "";
        while (n > 0) {
            int d = n % k;
            n = n / k;
            a += Integer.toString(d);
        }
        a = new StringBuilder(a).reverse().toString();
        a += "0";
        long c = 0;
        int pos = 0;
        int prev_pos = 0;
        while ((pos = a.indexOf('0', pos)) != -1) {
            String b = a.substring(prev_pos, pos);
            if (!b.isEmpty()) {
                c += Long.parseLong(b);
            }
            pos++;
            prev_pos = pos;
        }
        String ret = "";
        while (c > 0) {
            int d = (int) (c % k);
            c = c / k;
            ret += Integer.toString(d);
        }
        ret = new StringBuilder(ret).reverse().toString();
        return ret;
    }

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int n = scanner.nextInt();
        int k = scanner.nextInt();
        System.out.println(solution(n, k));
    }
}
