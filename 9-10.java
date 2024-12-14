import java.util.ArrayList;
import java.util.List;

public class Solution {
    public static int[] parking_time = new int[10000];
    public static int[] in_time = new int[10000];

    public static int get_num(String s, int mode) {
        if (mode == 0) {
            int h = (s.charAt(0) - '0') * 10 + (s.charAt(1) - '0');
            int m = (s.charAt(3) - '0') * 10 + (s.charAt(4) - '0');
            return h * 60 + m;
        } else if (mode == 1) {
            return (s.charAt(6) - '0') * 1000 + (s.charAt(7) - '0') * 100 + (s.charAt(8) - '0') * 10 + (s.charAt(9) - '0');
        } else if (mode == 2) {
            if (s.charAt(11) == 'I') return 0;
            return 1;
        }
        return 0;
    }

    public static int get_fee(int[] fees, int t) {
        int money = fees[1];
        if (fees[0] < t)
            money += (t - fees[0] + fees[2] - 1) / fees[2] * fees[3];
        return money;
    }

    public static int[] solution(int[] fees, String[] records) {
        List<Integer> answer = new ArrayList<>();
        for (int i = 0; i < 10000; i++)
            in_time[i] = -1;
        for (int i = 0; i < records.length; i++) {
            int t = get_num(records[i], 0);
            int c = get_num(records[i], 1);
            int d = get_num(records[i], 2);
            if (d == 0) {
                in_time[c] = t;
            } else {
                parking_time[c] += t - in_time[c];
                in_time[c] = -1;
            }
        }
        for (int i = 0; i < 10000; i++) {
            if (in_time[i] != -1) {
                parking_time[i] += 23 * 60 + 59 - in_time[i];
            }
        }
        for (int i = 0; i < 10000; i++) {
            if (parking_time[i] != 0) {
                answer.add(get_fee(fees, parking_time[i]));
            }
        }
        int[] ans = new int[answer.size()];
        for (int i = 0; i < ans.length; i++) ans[i] = answer.get(i);
        return ans;
    }
}
