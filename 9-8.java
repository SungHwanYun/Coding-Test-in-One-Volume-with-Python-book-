import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Solution {
    public static int[] solution(String[] id_list, String[] report, int k) {
        Map<String, Integer> mp = new HashMap<>();
        for (int i = 0; i < id_list.length; i++) {
            mp.put(id_list[i], i);
        }

        int N = id_list.length;
        int[][] report_result = new int[N][N];
        int[] bad_id = new int[N];

        for (int i = 0; i < report.length; i++) {
            String[] split = report[i].split(" ");
            String x = split[0];
            String y = split[1];
            int xidx = mp.get(x);
            int yidx = mp.get(y);
            report_result[xidx][yidx] = 1;
        }

        for (int j = 0; j < N; j++) {
            int cnt = 0;
            for (int i = 0; i < N; i++) {
                cnt += report_result[i][j];
            }
            if (cnt >= k) {
                bad_id[j] = 1;
            } else {
                bad_id[j] = 0;
            }
        }

        List<Integer> answer = new ArrayList<>();
        for (int i = 0; i < N; i++) {
            int cnt = 0;
            for (int j = 0; j < N; j++) {
                if (report_result[i][j] == 1 && bad_id[j] == 1) {
                    cnt++;
                }
            }
            answer.add(cnt);
        }
        int[] ans = new int[answer.size()];
        for (int i = 0; i < ans.length; i++) ans[i] = answer.get(i);
        return ans;
    }
}
