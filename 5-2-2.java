import java.util.*;

public class Main {
    static int upper_bound(List<Long> arr, long key) {
        int mid, N = arr.size();
        int low = 0;
        int high = N;
  
        while (low < high && low != N) {
            mid = low + (high - low) / 2;
            if (key >= arr.get(mid)) {
                low = mid + 1;
            }
            else {
                high = mid;
            }
        }
        return low;
    }

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int n = scanner.nextInt();
        int m = scanner.nextInt();
        List<Long> A = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            A.add(scanner.nextLong());
        }
        Collections.sort(A);
        while (m-- > 0) {
            long k = scanner.nextLong();
            int i = upper_bound(A, k);
            System.out.println(n - i);
        }
    }
}
