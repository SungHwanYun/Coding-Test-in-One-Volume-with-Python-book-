import java.util.*;

public class Main {
    static int lower_bound(List<Long> array, long key) {
        int low = 0, high = array.size();
        int mid;
        while (low < high) {
            mid = low + (high - low) / 2;
            if (key <= array.get(mid)) {
                high = mid;
            }
            else {
 
                low = mid + 1;
            }
        }
        if (low < array.size() && array.get(low) < key) {
            low++;
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
            int i = lower_bound(A, k);
            System.out.println(n - i);
        }
    }
}
