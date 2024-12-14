#include<bits/stdc++.h>
using namespace std;
int in_range(int r, int c) {
    return 0 <= r && r <= 4 && 0 <= c && c <= 4;
}
int main() {
    int A[5][5];
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            cin >> A[i][j];
        }
    }

    int r, c; cin >> r >> c;
    int dr[4] = { -1,1,0,0 }, dc[4] = { 0,0,-1,1 };
    for (int i = 0; i < 4; i++) {
        int nr = r + dr[i], nc = c + dc[i];
        if (in_range(nr, nc) && A[nr][nc] == 1) {
            cout << 1; exit(0);
        }
    }
    cout << 0;
}