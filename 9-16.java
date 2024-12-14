import java.util.*;

public class Solution {
    static Map<String, Integer>[] mp;

    public static String[] solution(String[] orders, int[] course) {
        List<String> answer = new ArrayList<>();
        mp = new HashMap[12];
        for (int i = 0; i < 12; i++) {
            mp[i] = new HashMap<>();
        }
        for (int i = 0; i < orders.length; i++) {
            char[] chars = orders[i].toCharArray();
            Arrays.sort(chars);
            orders[i] = new String(chars);
        }
        for (int i = 0; i < orders.length; i++) {
            for (int j = 0; j < course.length; j++) {
                String s = "";
                build_menu(orders[i], 0, course[j], s);
            }
        }
        for (int i = 0; i < course.length; i++) {
            int len = course[i];
            int mx = -1;
            for (Map.Entry<String, Integer> entry : mp[len].entrySet()) {
                mx = Math.max(mx, entry.getValue());
            }
            if (mx < 2) continue;
            for (Map.Entry<String, Integer> entry : mp[len].entrySet()) {
                if (entry.getValue() == mx)
                    answer.add(entry.getKey());
            }
        }
        Collections.sort(answer);
        String[] ans = new String[answer.size()];
        for (int i = 0; i < ans.length; i++) ans[i] = answer.get(i);
        return ans;
    }

    public static void build_menu(String src, int idx, int cnt, String dst) {
        int n = src.length();
        int remain = n - idx;
        if (remain < cnt) return;
        if (idx == n || cnt == 0) {
            if (cnt == 0) {
                if (!mp[dst.length()].containsKey(dst))
                    mp[dst.length()].put(dst, 1);
                else
                    mp[dst.length()].put(dst, mp[dst.length()].get(dst) + 1);
            }
            return;
        }
        build_menu(src, idx + 1, cnt, dst);
        dst += src.charAt(idx);
        build_menu(src, idx + 1, cnt - 1, dst);
    }
}