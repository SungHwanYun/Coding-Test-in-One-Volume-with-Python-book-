#include <bits/stdc++.h>
using namespace std;

long long solution(int cap, int n, vector<int> deliveries, vector<int> pickups) {
    int i = n - 1;
    int j = n - 1;
    long long answer = 0;
    while (i >= 0 || j >= 0) {
        int x = cap;
        int ii = -1;
        while (i >= 0 && x > 0) {
            if (ii == -1 && deliveries[i] > 0) {
                ii = i;
            }
            int y = min(deliveries[i], x);
            x -= y;
            deliveries[i] -= y;
            if (deliveries[i] == 0) {
                i -= 1;
            }
        }
        x = cap;
        int jj = -1;
        while (j >= 0 && x > 0) {
            if (jj == -1 && pickups[j] > 0) {
                jj = j;
            }
            int y = min(pickups[j], x);
            x -= y;
            pickups[j] -= y;
            if (pickups[j] == 0) {
                j -= 1;
            }
        }
        if (ii == -1 && jj == -1) {
            break;
        }
        answer += max(ii + 1, jj + 1) * 2;
    }
    return answer;
}