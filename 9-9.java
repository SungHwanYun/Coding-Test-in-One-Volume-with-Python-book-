import java.util.*;

public class Solution {
    public static boolean isPrime(long x) {
        if (x <= 1) return false;
        for (long i = 2; i * i <= x; i++)
            if (x % i == 0) return false;
        return true;
    }

    public int solution(int n, int k) {
        int answer = 0;
        while (n > 0) {
            long P = 0, x = 1;
            while (n > 0) {
                int d = n % k;
                n = n / k;
                if (d == 0) break;
                P += d * x;
                x *= 10;
            }
            if (isPrime(P))
                answer++;
        }
        return answer;
    }
}
