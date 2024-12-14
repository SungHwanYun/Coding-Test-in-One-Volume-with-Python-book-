import java.util.Arrays;
import java.util.List;

public class Solution {
    static int[] A = new int[360004];
    static long[] B = new long[360004];
    static int P, T;

    public static int convert_time(String str) {
        int h = Integer.parseInt(str.substring(0, 2));
        int m = Integer.parseInt(str.substring(3, 5));
        int s = Integer.parseInt(str.substring(6, 8));
        return h * 3600 + m * 60 + s;
    }

    public static String solution(String play_time, String adv_time, String[] logs) {
        P = convert_time(play_time);
        T = convert_time(adv_time);
        for (int i = 0; i < logs.length; i++) {
            int s = convert_time(logs[i]);
            int e = convert_time(logs[i].substring(9, 17));
            A[s]++;
            A[e]--;
        }
        for (int i = 1; i <= P; i++)
            A[i] += A[i - 1];
        B[0] = A[0];
        for (int i = 1; i <= P; i++)
            B[i] = B[i - 1] + A[i];
        int x = 0;
        long y = B[T - 1];
        for (int i = 1; i + T <= P; i++) {
            long sum = B[i + T - 1] - B[i - 1];
            if (sum > y) {
                y = sum;
                x = i;
            }
        }
        int h = x / 3600;
        int m = (x - h * 3600) / 60;
        int s = x % 60;
        String answer = String.format("%02d:%02d:%02d", h, m, s);
        return answer;
    }
}