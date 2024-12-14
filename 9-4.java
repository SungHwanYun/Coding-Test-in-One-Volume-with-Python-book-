import java.util.ArrayList;
import java.util.List;
import java.util.Collections;

public class Solution {
    public static List<Integer> translate_binary(long n) {
        List<Integer> answer = new ArrayList<>();
        while (n > 0) {
            int x = (int)(n % 2);
            n = n / 2;
            answer.add(x);
        }
        long y = 1;
        while (y <= answer.size()) {
            y = y * 2;
        }
        while (answer.size() + 1 < y) {
            answer.add(0);
        }
        Collections.reverse(answer);
        return answer;
    }
    
    public static int solve(List<Integer> b, int st, int en) {
        if (st == en) {
            return 1;
        }
        int r = (en + st) / 2;
        if (b.get(r) == 0) {
            for (int i = st; i <= en; i++) {
                if (b.get(i) == 1) {
                    return 0;
                }
            }
            return 1;
        }
        else {
            return solve(b, st, r - 1) & solve(b, r + 1, en);
        }
    }
    
    public static int[] solution(long[] numbers) {
        List<Integer> answer = new ArrayList<>();
        for (long n : numbers) {
            List<Integer> b = translate_binary(n);
            answer.add(solve(b, 0, b.size() - 1));
        }
        int[] ans = new int[answer.size()];
        for (int i = 0; i < ans.length; i++) ans[i] = answer.get(i);
        return ans;
    }
}