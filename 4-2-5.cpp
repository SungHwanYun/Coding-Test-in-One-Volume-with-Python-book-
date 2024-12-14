#include<bits/stdc++.h>
using namespace std;
int in_range(int loc[2]) {
    return 0 <= loc[0] && loc[0] <= 4 && 0 <= loc[1] && loc[1] <= 4;
}
int solve(int A[5][5], int aloc[2], int apple_num) {
    if (apple_num == 0) return 0;
    int ret = -1;
    int dr[4] = { -1,1,0,0 }, dc[4] = { 0,0,-1,1 };
    for (int d = 0; d < 4; d++) {
        int nloc[2] = { aloc[0] + dr[d], aloc[1] + dc[d] };
        if (in_range(nloc) && A[nloc[0]][nloc[1]] != -1) {
            int prv_value = A[aloc[0]][aloc[1]];
            A[aloc[0]][aloc[1]] = -1;
            int cur_ret = solve(A, nloc, apple_num - A[nloc[0]][nloc[1]]);
            if (cur_ret != -1) {
                cur_ret++;
            }
            if (cur_ret != -1) {
                if (ret == -1 || cur_ret < ret) {
                    ret = cur_ret;
                }
            }
            A[aloc[0]][aloc[1]] = prv_value;
        }
    }
    return ret;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(NULL);
    cout.tie(NULL);

    int A[5][5];
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            cin >> A[i][j];
        }
    }
    int aloc[2]; cin >> aloc[0] >> aloc[1];
    cout << solve(A, aloc, 3);
}