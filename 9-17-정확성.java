import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Solution {
    public static Map<String, Integer> mp = new HashMap<>();
    public static List<List<Integer>> applicant = new ArrayList<>();

    public static List<Integer> split_info(String str) {
        int x, y;
        x = str.indexOf(' ');
        String lang = str.substring(0, x);
        y = str.indexOf(' ', x + 1);
        String job = str.substring(x + 1, y);
        x = y;
        y = str.indexOf(' ', x + 1);
        String car = str.substring(x + 1, y);
        x = y;
        y = str.indexOf(' ', x + 1);
        String food = str.substring(x + 1, y);
        int point = Integer.parseInt(str.substring(y + 1));
        List<Integer> ret = new ArrayList<>();
        ret.add(mp.get(lang));
        ret.add(mp.get(job));
        ret.add(mp.get(car));
        ret.add(mp.get(food));
        ret.add(point);
        return ret;
    }

    public static List<Integer> split_query(String str) {
        int x, y;
        x = str.indexOf(' ');
        String lang = str.substring(0, x);
        x = str.indexOf(' ', x + 1);
        y = str.indexOf(' ', x + 1);
        String job = str.substring(x + 1, y);
        x = str.indexOf(' ', y + 1);
        y = str.indexOf(' ', x + 1);
        String car = str.substring(x + 1, y);
        x = str.indexOf(' ', y + 1);
        y = str.indexOf(' ', x + 1);
        String food = str.substring(x + 1, y);
        int point = Integer.parseInt(str.substring(y + 1));
        List<Integer> ret = new ArrayList<>();
        ret.add(mp.get(lang));
        ret.add(mp.get(job));
        ret.add(mp.get(car));
        ret.add(mp.get(food));
        ret.add(point);
        return ret;
    }

    public static boolean is_ok(List<Integer> qry, List<Integer> app) {
        for (int i = 0; i < 4; i++) {
            if (qry.get(i) != 0 && qry.get(i) != app.get(i)) return false;
        }
        return true;
    }

    public static int[] solution(String[] info, String[] query) {
        mp.put("-", 0);
        mp.put("cpp", 1);
        mp.put("java", 2);
        mp.put("python", 3);
        mp.put("backend", 1);
        mp.put("frontend", 2);
        mp.put("junior", 1);
        mp.put("senior", 2);
        mp.put("chicken", 1);
        mp.put("pizza", 2);

        for (int i = 0; i < info.length; i++) {
            applicant.add(split_info(info[i]));
        }

        List<Integer> answer = new ArrayList<>();
        for (int i = 0; i < query.length; i++) {
            List<Integer> v = split_query(query[i]);
            int cnt = 0;
            for (int j = 0; j < applicant.size(); j++) {
                if (is_ok(v, applicant.get(j)) && v.get(4) <= applicant.get(j).get(4))
                    cnt++;
            }
            answer.add(cnt);
        }
        int[] ans = new int[answer.size()];
        for (int i = 0; i < ans.length; i++) ans[i] = answer.get(i);
        return ans;
    }
}
