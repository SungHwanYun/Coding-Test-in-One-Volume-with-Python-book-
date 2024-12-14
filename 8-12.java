import java.util.*;

public class Main {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();
        List<List<String>> A = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            List<String> temp = new ArrayList<>();
            temp.add(sc.next());
            temp.add(sc.next());
            temp.add(sc.next());
            A.add(temp);
        }
        List<Long> T = new ArrayList<>(Collections.nCopies(24 * 60 * 60, 0L));
        List<Long> R = new ArrayList<>(Collections.nCopies(24 * 60 * 60, 0L));
        List<Long> answer = new ArrayList<>();
        boolean flag = false;
        for (int i = 0; i < n; i++) {
            if (A.get(i).get(0).equals("1")) {
                int start = Integer.parseInt(A.get(i).get(1).substring(0, 2)) * 3600 + Integer.parseInt(A.get(i).get(1).substring(3, 5)) * 60 + Integer.parseInt(A.get(i).get(1).substring(6));
                int end = Integer.parseInt(A.get(i).get(2).substring(0, 2)) * 3600 + Integer.parseInt(A.get(i).get(2).substring(3, 5)) * 60 + Integer.parseInt(A.get(i).get(2).substring(6));
                T.set(start, T.get(start) + 1);
                T.set(end, T.get(end) - 1);
            } else {
                if (!flag) {
                    for (int j = 1; j < 24 * 60 * 60; j++) {
                        T.set(j, T.get(j) + T.get(j - 1));
                    }
                    flag = true;
                    R.set(0, T.get(0));
                    for (int j = 1; j < 24 * 60 * 60; j++) {
                        R.set(j, R.get(j - 1) + T.get(j));
                    }
                }
                int start = Integer.parseInt(A.get(i).get(1).substring(0, 2)) * 3600 + Integer.parseInt(A.get(i).get(1).substring(3, 5)) * 60 + Integer.parseInt(A.get(i).get(1).substring(6));
                int end = Integer.parseInt(A.get(i).get(2).substring(0, 2)) * 3600 + Integer.parseInt(A.get(i).get(2).substring(3, 5)) * 60 + Integer.parseInt(A.get(i).get(2).substring(6));
                long ret = R.get(end - 1);
                if (start != 0) {
                    ret -= R.get(start - 1);
                }
                answer.add(ret);
            }
        }
        for (int i = 0; i < answer.size(); i++) {
            System.out.println(answer.get(i));
        }
    }
}
