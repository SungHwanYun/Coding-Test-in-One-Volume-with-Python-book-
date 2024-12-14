import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Solution {
    public static int[] solution(String[] info, String[] query) {
        Map<String, Integer> mp = new HashMap<>();
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

        List<Integer>[][][][] score = new ArrayList[3][3][3][4];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    for (int l = 0; l < 4; l++) {
                        score[i][j][k][l] = new ArrayList<>();
                    }
                }
            }
        }

        for (String str : info) {
            List<Integer> v = splitInfo(str, mp);
            for (int lang = 0; lang <= 3; lang++) {
                for (int job = 0; job <= 2; job++) {
                    for (int career = 0; career <= 2; career++) {
                        for (int food = 0; food <= 2; food++) {
                            List<Integer> v2 = new ArrayList<>();
                            v2.add(lang);
                            v2.add(job);
                            v2.add(career);
                            v2.add(food);
                            if (isOk(v2, v)) {
                                score[food][career][job][lang].add(v.get(4));
                            }
                        }
                    }
                }
            }
        }

        for (int lang = 0; lang <= 3; lang++) {
            for (int job = 0; job <= 2; job++) {
                for (int career = 0; career <= 2; career++) {
                    for (int food = 0; food <= 2; food++) {
                        score[food][career][job][lang].sort(null);
                    }
                }
            }
        }

        List<Integer> answer = new ArrayList<>();
        for (String str : query) {
            List<Integer> v = splitQuery(str, mp);
            int lang = v.get(0);
            int job = v.get(1);
            int car = v.get(2);
            int food = v.get(3);
            int point = v.get(4);
            int len = score[food][car][job][lang].size() - lowerBound(score[food][car][job][lang], point);
            answer.add(len);
        }
        int[] ans = new int[answer.size()];
        for (int i = 0; i < ans.length; i++) ans[i] = answer.get(i);
        return ans;
    }

    private static List<Integer> splitInfo(String str, Map<String, Integer> mp) {
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

    private static List<Integer> splitQuery(String str, Map<String, Integer> mp) {
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

    private static boolean isOk(List<Integer> qry, List<Integer> app) {
        for (int i = 0; i < 4; i++) {
            if (qry.get(i) != 0 && qry.get(i) != app.get(i)) {
                return false;
            }
        }
        return true;
    }

    private static int lowerBound(List<Integer> list, int target) {
        int left = 0;
        int right = list.size();
        while (left < right) {
            int mid = (left + right) / 2;
            if (list.get(mid) >= target) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return left;
    }
}
