import java.util.*;

public class Main {
    public static int solution(ArrayList<ArrayList<Integer>> A) {
        int[] v = new int[6];
        for (int i=0; i<6; i++) v[i]=i;
        int answer = (int)1e8;
        do {
            ArrayList<ArrayList<Integer>> D = new ArrayList<>();
            for (int i = 0; i < 6; i++) {
                D.add(new ArrayList<>(Arrays.asList(0, 0)));
            }
            D.get(0).set(0, A.get(v[0] * 2).get(v[0] * 2 + 1));
            D.get(0).set(1, A.get(v[0] * 2).get(v[0] * 2 + 1));
            for (int i = 1; i < 6; i++) {
                D.get(i).set(0, Math.min(D.get(i - 1).get(0) + A.get(v[i - 1] * 2).get(v[i] * 2 + 1) + A.get(v[i] * 2).get(v[i] * 2 + 1),
                    D.get(i - 1).get(1) + A.get(v[i-1] * 2 + 1).get(v[i] * 2 + 1) + A.get(v[i] * 2).get(v[i] * 2 + 1)));
                D.get(i).set(1, Math.min(D.get(i - 1).get(0) + A.get(v[i-1] * 2).get(v[i] * 2) + A.get(v[i] * 2).get(v[i] * 2 + 1),
                    D.get(i - 1).get(1) + A.get(v[i-1] * 2 + 1).get(v[i] * 2) + A.get(v[i] * 2).get(v[i] * 2 + 1)));
            }
            answer = Math.min(answer, Math.min(D.get(5).get(0), D.get(5).get(1)));
        } while (next_permutation(v));
        return answer;
    }

    public static boolean next_permutation(int[] v) {
        int n = v.length;
        int i = n - 1;
        while (i > 0 && v[i - 1] >= v[i]) i--;
        if (i <= 0) return false;
        int j = n - 1;
        while (v[j] <= v[i - 1]) j--;
        int temp = v[i-1];
        v[i-1]=v[j];
        v[j]=temp;
        j = n - 1;
        while (i < j) {
            temp = v[i];
            v[i] = v[j];
            v[j] = temp;
            i++;
            j--;
        }
        return true;
    }    

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        ArrayList<ArrayList<Integer>> A = new ArrayList<>();
        for (int i = 0; i < 12; i++) {
            ArrayList<Integer> row = new ArrayList<>();
            for (int j = 0; j < 12; j++) {
                row.add(scanner.nextInt());
            }
            A.add(row);
        }
        System.out.println(solution(A));
    }
}
