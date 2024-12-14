import java.util.Arrays;
import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        String A = scanner.next();
        char[] arr = A.toCharArray();
        Arrays.sort(arr);
        System.out.println(new String(arr));
    }
}
