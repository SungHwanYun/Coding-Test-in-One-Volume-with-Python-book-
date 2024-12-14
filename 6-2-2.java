import java.util.*;

public class Main {
    static class Pair {
        int first;
        int second;

        Pair(int first, int second) {
            this.first = first;
            this.second = second;
        }
    }

    static class Node {
        int vertex;
        long distance;

        Node(int vertex, long distance) {
            this.vertex = vertex;
            this.distance = distance;
        }
    }

    static final long INF = (long) 1e18;

    static int N, X, Y, Z;
    static List<Pair>[] E;
    static boolean[] selected;

    static void dijkstra(int start, int end, List<Long> dist) {
        dist.set(start, 0L);
        PriorityQueue<Node> pq = new PriorityQueue<>((a, b) -> Long.compare(a.distance, b.distance));
        pq.add(new Node(start, 0L));
        while (!pq.isEmpty()) {
            Node node = pq.poll();
            long cost = node.distance;
            int here = node.vertex;
            if (dist.get(here) < cost) continue;
            for (Pair edge : E[here]) {
                int there = edge.first;
                long nextDist = cost + edge.second;
                if (dist.get(there) > nextDist) {
                    dist.set(there, nextDist);
                    pq.add(new Node(there, nextDist));
                }
            }
        }
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int q, u, v, x, i;
        N = sc.nextInt();
        q = sc.nextInt();
        E = new ArrayList[N + 1];
        for (i = 1; i <= N; i++) E[i] = new ArrayList<>();
        while (q-- > 0) {
            u = sc.nextInt();
            v = sc.nextInt();
            x = sc.nextInt();
            E[u].add(new Pair(v, x));
            E[v].add(new Pair(u, x));
        }
        X = sc.nextInt();
        Z = sc.nextInt();
        q = sc.nextInt();
        List<Integer> P = new ArrayList<>();
        for (i = 0; i < q; i++) P.add(sc.nextInt());
        List<Long> dist_X = new ArrayList<>(Collections.nCopies(N + 1, INF));
        selected = new boolean[N + 1];
        dijkstra(X, Z, dist_X);
        List<Long> dist_Z = new ArrayList<>(Collections.nCopies(N + 1, INF));
        selected = new boolean[N + 1];
        dijkstra(Z, X, dist_Z);
        long ans = INF;
        for (int p : P) {
            if (dist_X.get(p) < INF && dist_Z.get(p) < INF && ans > dist_X.get(p) + dist_Z.get(p))
                ans = dist_X.get(p) + dist_Z.get(p);
        }
        if (ans >= INF) ans = -1;
        System.out.println(ans);
    }
}
