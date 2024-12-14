#include<bits/stdc++.h>
using namespace std;
int in_range(int loc[2]) {
    return 0 <= loc[0] && loc[0] <= 4 && 0 <= loc[1] && loc[1] <= 4;
}
int get_apple(int A[5][5], int aloc[2], int iloc[2], int jloc[2], int kloc[2]) {
    int apple_num = 0;
    if (!in_range(iloc) || !in_range(jloc)) return 0;
    if (A[iloc[0]][iloc[1]] == -1 || A[jloc[0]][jloc[1]] == -1) return 0;
    if (aloc[0] == jloc[0] && aloc[1] == jloc[1]) return 0;
    apple_num = A[iloc[0]][iloc[1]] + A[jloc[0]][jloc[1]];
    if (in_range(kloc) && A[kloc[0]][kloc[1]] == 1 &&
        (iloc[0] != kloc[0] || iloc[1] != kloc[1]))
        apple_num += 1;
    return apple_num;
}
int main() {
    ios::sync_with_stdio(false);
    cin.tie(NULL);
    cout.tie(NULL);

    int n = 5, A[5][5];
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            cin >> A[i][j];
        }
    }
    int aloc[2]; cin >> aloc[0] >> aloc[1];

    int dr[4] = { -1,1,0,0 }, dc[4] = { 0,0,-1,1 };
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 4; k++) {
                int iloc[2] = { aloc[0] + dr[i], aloc[1] + dc[i] };
                int jloc[2] = { iloc[0] + dr[j], iloc[1] + dc[j] };
                int kloc[2] = { jloc[0] + dr[k], jloc[1] + dc[k] };
                if (get_apple(A, aloc, iloc, jloc, kloc) >= 2) {
                    cout << 1; exit(0);
                }
            }
        }
    }
    cout << 0;
}