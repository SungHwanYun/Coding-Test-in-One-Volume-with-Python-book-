import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        String A = scanner.next();
        int k = scanner.nextInt();
        int n = A.length();
        List<Integer> perm = new ArrayList<>(Collections.nCopies(n, 0));
        for (int i = n - k; i < n; i++) {
            perm.set(i, 1);
        }
        List<String> C = new ArrayList<>();
        do {
            StringBuilder s = new StringBuilder();
            for (int i = 0; i < n; i++) {
                if (perm.get(i) == 1) {
                    s.append(A.charAt(i));
                }
            }
            C.add(s.toString());
        } while (nextPermutation(perm));
        Collections.sort(C);
        for (String c : C) {
            System.out.println(c);
        }
    }

    private static boolean nextPermutation(List<Integer> perm) {
        int n = perm.size();
        int i = n - 2;
        while (i >= 0 && perm.get(i) >= perm.get(i + 1)) {
            i--;
        }
        if (i < 0) {
            return false;
        }
        int j = n - 1;
        while (perm.get(i) >= perm.get(j)) {
            j--;
        }
        Collections.swap(perm, i, j);
        int left = i + 1;
        int right = n - 1;
        while (left < right) {
            Collections.swap(perm, left, right);
            left++;
            right--;
        }
        return true;
    }
}
