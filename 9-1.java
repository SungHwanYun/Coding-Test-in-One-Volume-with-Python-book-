import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.ArrayList;

public class Solution {
    public static int translateDays(String x) {
        return Integer.parseInt(x.substring(0, 4)) * 12 * 28 + Integer.parseInt(x.substring(5, 7)) * 28 + Integer.parseInt(x.substring(8));
    }
    
    public int[] solution(String today, String[] terms, String[] privacies) {
        Map<String, Integer> T = new HashMap<>();
        for (String t : terms) {
            String[] parts = t.split(" ");
            String x = parts[0];
            String y = parts[1];
            T.put(x, Integer.parseInt(y) * 28);
        }
        List<Integer> answer = new ArrayList<>();
        int todayDays = translateDays(today);
        for (int i = 0; i < privacies.length; i++) {
            String[] parts = privacies[i].split(" ");
            String x = parts[0];
            String y = parts[1];
            int a = translateDays(x) + T.get(y);
            if (a <= todayDays) {
                answer.add(i + 1);
            }
        }
        int[] ans = new int[answer.size()];
        for (int i=0; i<ans.length; i++) ans[i]=answer.get(i);
        return ans;
    }
}
