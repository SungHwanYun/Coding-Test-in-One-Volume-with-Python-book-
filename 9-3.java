import java.util.ArrayList;
import java.util.List;

class Solution {
    public static List<Integer> get_info(int[] user, int[] emoticons, List<Integer> x) {
        int m = emoticons.length;
        int money = 0;
        for (int i = 0; i < m; i++) {
            if (x.get(i) >= user[0]) {
                money += emoticons[i] * (100 - x.get(i)) / 100;
            }
        }
        if (money >= user[1]) {
            return List.of(1, 0);
        } else {
            return List.of(0, money);
        }
    }

    public int[] solution(int[][] users, int[] emoticons) {
        int n = users.length;
        int m = emoticons.length;
        List<Integer> answer = new ArrayList<>();
        answer.add(0); answer.add(0);
        for (int k = 0; k < (1 << (2 * m)); k++) {
            List<Integer> x = new ArrayList<>();
            for (int i = 0; i < m; i++) {
                int a = (k >> (i * 2)) & 0x3;
                x.add((a + 1) * 10);
            }
            List<Integer> ans = new ArrayList<>();
            ans.add(0); ans.add(0);
            for (int i = 0; i < n; i++) {
                List<Integer> ret = get_info(users[i], emoticons, x);
                ans.set(0, ans.get(0) + ret.get(0));
                ans.set(1, ans.get(1) + ret.get(1));
            }
            if (answer.get(0) < ans.get(0) || (answer.get(0) == ans.get(0) && answer.get(1) < ans.get(1))) {
                answer.set(0, ans.get(0));
                answer.set(1, ans.get(1));
            }
        }
        int[] ret = new int[answer.size()];
        for (int i = 0; i < ret.length; i++) {
            ret[i] = answer.get(i);
        }
        return ret;
    }
}
