import java.util.ArrayList;
import java.util.List;

class Solution {
    static List<List<String>> board = new ArrayList<>(51);
    static List<List<List<Integer>>> P = new ArrayList<>(51);

    static List<Integer> do_find(List<Integer> x) {
        if (P.get(x.get(0)).get(x.get(1)).equals(x)) {
            return x;
        }
        P.get(x.get(0)).set(x.get(1), do_find(P.get(x.get(0)).get(x.get(1))));
        return P.get(x.get(0)).get(x.get(1));
    }

    static void do_merge(List<Integer> x, List<Integer> y) {
        List<Integer> px = do_find(x);
        List<Integer> py = do_find(y);
        P.get(py.get(0)).set(py.get(1), px);
    }

    static String[] solution(String[] commands) {
        for (int i = 0; i <= 50; i++) {
            List<String> row = new ArrayList<>(51);
            List<List<Integer>> pRow = new ArrayList<>(51);
            for (int j = 0; j <= 50; j++) {
                row.add("");
                List<Integer> p = new ArrayList<>(2);
                p.add(i);
                p.add(j);
                pRow.add(p);
            }
            board.add(row);
            P.add(pRow);
        }

        List<String> answer = new ArrayList<>();
        for (String cc : commands) {
            List<String> cmd = new ArrayList<>();
            String word = "";
            for (char ch : cc.toCharArray()) {
                if (ch == ' ') {
                    cmd.add(word);
                    word = "";
                } else {
                    word += ch;
                }
            }
            cmd.add(word);
            if (cmd.get(0).equals("UPDATE") && cmd.size() == 4) {
                List<Integer> x = new ArrayList<>();
                x.add(Integer.parseInt(cmd.get(1)));
                x.add(Integer.parseInt(cmd.get(2)));
                List<Integer> px = do_find(x);
                board.get(px.get(0)).set(px.get(1), cmd.get(3));
            } else if (cmd.get(0).equals("UPDATE") && cmd.size() == 3) {
                for (int r = 1; r <= 50; r++) {
                    for (int c = 1; c <= 50; c++) {
                        if (board.get(r).get(c).equals(cmd.get(1))) {
                            board.get(r).set(c, cmd.get(2));
                        }
                    }
                }
            } else if (cmd.get(0).equals("MERGE")) {
                List<Integer> x = new ArrayList<>();
                x.add(Integer.parseInt(cmd.get(1)));
                x.add(Integer.parseInt(cmd.get(2)));
                List<Integer> y = new ArrayList<>();
                y.add(Integer.parseInt(cmd.get(3)));
                y.add(Integer.parseInt(cmd.get(4)));
                if (x.equals(y)) {
                    continue;
                }
                List<Integer> px = do_find(x);
                List<Integer> py = do_find(y);
                String value = "";
                if (board.get(px.get(0)).get(px.get(1)).equals("")) {
                    value = board.get(py.get(0)).get(py.get(1));
                } else {
                    value = board.get(px.get(0)).get(px.get(1));
                }
                board.get(px.get(0)).set(px.get(1), "");
                board.get(py.get(0)).set(py.get(1), "");
                do_merge(px, py);
                board.get(px.get(0)).set(px.get(1), value);
            } else if (cmd.get(0).equals("UNMERGE")) {
                List<Integer> x = new ArrayList<>();
                x.add(Integer.parseInt(cmd.get(1)));
                x.add(Integer.parseInt(cmd.get(2)));
                List<Integer> px = do_find(x);
                String ss = board.get(px.get(0)).get(px.get(1));
                List<List<Integer>> L = new ArrayList<>();
                for (int r = 1; r <= 50; r++) {
                    for (int c = 1; c <= 50; c++) {
                        List<Integer> y = do_find(List.of(r, c));
                        if (y.equals(px)) {
                            L.add(List.of(r, c));
                        }
                    }
                }
                for (List<Integer> rc : L) {
                    P.get(rc.get(0)).set(rc.get(1), List.of(rc.get(0), rc.get(1)));
                    board.get(rc.get(0)).set(rc.get(1), "");
                }
                board.get(x.get(0)).set(x.get(1), ss);
            } else {
                List<Integer> x = new ArrayList<>();
                x.add(Integer.parseInt(cmd.get(1)));
                x.add(Integer.parseInt(cmd.get(2)));
                List<Integer> px = do_find(x);
                if (board.get(px.get(0)).get(px.get(1)).equals("")) {
                    answer.add("EMPTY");
                } else {
                    answer.add(board.get(px.get(0)).get(px.get(1)));
                }
            }
        }
        String[] ans = new String[answer.size()];
        for (int i = 0; i < ans.length; i++) ans[i] = answer.get(i);
        return ans;
    }
}
