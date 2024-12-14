#include<bits/stdc++.h>
using namespace std;
int main() {
    ios::sync_with_stdio(false);
    cin.tie(NULL);
    cout.tie(NULL);
    string S;
    cin >> S;
    string T;
    int i = 0;
    while (i < S.length()) {
        if (S[i] != 'a' && S[i] != 'A') {
            T = T + S[i];
            i++;
            continue;
        }

        int j = i + 1;
        while (j < S.length()) {
            if (S[j] != 'a' && S[j] != 'A') {
                break;
            }
            j++;
        }
        if (j - i == 1) {
            T = T + S[i];
        }
        else {
            T = T + 'a';
        }
        i = j;
    }
    cout << T;
}