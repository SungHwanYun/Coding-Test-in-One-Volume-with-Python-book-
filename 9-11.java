import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Solution {
    static List<Integer> answer;
    static int point = 0;

    public static int[] solution(int n, int[] info) {
        List<Integer> rian = new ArrayList<>();
        answer = new ArrayList<>();
        answer.add(-1);
        for (int i = 0, j = info.length - 1; i < j; i++, j--) {
            int t = info[i];
            info[i] = info[j];
            info[j] = t;
        }
        solve(rian, n, info);
        Collections.reverse(answer);
        int[] ans = new int[answer.size()];
        for (int i = 0; i < ans.length;i++) ans[i] = answer.get(i);
        return ans;
    }

    public static int get_point(List<Integer> rian, int[] info) {
        int rian_point = 0, apeach_point = 0;
        for (int i = 1; i <= 10; i++) {
            if (rian.get(i) == 0 && info[i] == 0)
                continue;
            if (rian.get(i) > info[i])
                rian_point += i;
            else
                apeach_point += i;
        }
        if (rian_point > apeach_point)
            return rian_point - apeach_point;
        else
            return -1;
    }

    public static void solve(List<Integer> rian, int n, int[] info) {
        int arrow_sum = 0, remained_arrow;
        for (int i = 0; i < rian.size(); i++)
            arrow_sum += rian.get(i);
        remained_arrow = n - arrow_sum;
        if (rian.size() == 10) {
            rian.add(remained_arrow);
            int x = get_point(rian, info);
            if (x > 0 && x > point) {
                point = x;
                answer = new ArrayList<>(rian);
            }
            rian.remove(rian.size() - 1);
            return;
        }
        for (int i = remained_arrow; i >= 0; --i) {
            rian.add(i);
            solve(rian, n, info);
            rian.remove(rian.size() - 1);
        }
    }
}