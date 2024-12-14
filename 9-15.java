import java.util.*;

public class Solution {
    public static String solution(String new_id) {
        int i;
        String answer = "";
        for (i = 0; i < new_id.length(); i++) {
            if ('A' <= new_id.charAt(i) && new_id.charAt(i) <= 'Z')
                new_id = new_id.substring(0, i) + (char) (new_id.charAt(i) - 'A' + 'a') + new_id.substring(i + 1);
        }
        for (i = 0; i < new_id.length(); i++) {
            if (('a' <= new_id.charAt(i) && new_id.charAt(i) <= 'z') || ('0' <= new_id.charAt(i) && new_id.charAt(i) <= '9') || new_id.charAt(i) == '-' || new_id.charAt(i) == '_' ||
                    new_id.charAt(i) == '.') {
                answer += new_id.charAt(i);
            }
        }
        new_id = answer;
        answer = "";
        if (new_id.length() > 0)
            answer += new_id.charAt(0);
        for (i = 1; i < new_id.length(); i++) {
            if (new_id.charAt(i - 1) != '.' || new_id.charAt(i) != '.') {
                answer += new_id.charAt(i);
            }
        }
        new_id = answer;
        answer = "";
        if (new_id.length() > 0 && new_id.charAt(0) != '.')
            answer += new_id.charAt(0);
        for (i = 1; i < new_id.length() - 1; i++)
            answer += new_id.charAt(i);
        if (new_id.length() > 1 && new_id.charAt(new_id.length() - 1) != '.')
            answer += new_id.charAt(new_id.length() - 1);
        new_id = answer;
        if (new_id.length() == 0)
            new_id += 'a';
        if (new_id.length() > 15) {
            new_id = new_id.substring(0, 15);
            if (new_id.charAt(14) == '.')
                new_id = new_id.substring(0, 14);
        }
        while (new_id.length() <= 2)
            new_id += new_id.charAt(new_id.length() - 1);
        answer = new_id;
        return answer;
    }
}
