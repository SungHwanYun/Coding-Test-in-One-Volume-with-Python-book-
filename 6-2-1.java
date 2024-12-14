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
    
    static class Node implements Comparable<Node> {
        int index;
        long distance;
        
        Node(int index, long distance) {
            this.index = index;
            this.distance = distance;
        }
        
        public int compareTo(Node other) {
            return Long.compare(this.distance, other.distance);
        }
    }
    
    static final long INF = (long)1e18;
    static List<Pair>[] E;
    
    static void dijkstra(int start, int end, List<Long> dist) {
        dist.set(start, 0L);
        PriorityQueue<Node> pq = new PriorityQueue<>();
        pq.add(new Node(start, 0L));
        
        while (!pq.isEmpty()) {
            Node node = pq.poll();
            long cost = node.distance;
            int here = node.index;
            
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
        int N = sc.nextInt();
        int M = sc.nextInt();
        int X, Z;
        int u, v, w;
        
        E = new ArrayList[N + 1];
        for (int i = 1; i <= N; i++) {
            E[i] = new ArrayList<>();
        }
        
        for (int i = 0; i < M; i++) {
            u = sc.nextInt();
            v = sc.nextInt();
            w = sc.nextInt();
            E[u].add(new Pair(v, w));
        }
        
        X = sc.nextInt();
        Z = sc.nextInt();
        
        List<Long> dist = new ArrayList<>(N + 1);
        for (int i = 0; i <= N; i++) {
            dist.add(INF);
        }
        
        dijkstra(X, Z, dist);
        
        if (dist.get(Z) >= INF) {
            dist.set(Z, -1L);
        }
        
        System.out.println(dist.get(Z));
    }
}
