#include <bits/stdc++.h>
using namespace std;
int N;
int P(vector<int>& X) {
    int ret = 1;
    for (int x : X) {
        ret *= x;
    }
    return ret;
}

int get_joined_num(vector<int>& X) {
    string x = "";
    for (int num : X) {
        x += to_string(num);
    }
    return stoi(x);
}

vector<int> solve(int n, vector<int>& A, vector<int>& B) {
    vector<int> ret = { -1 };
    if (n == 0) {
        int pa = P(A), pb = P(B);
        if (pa < pb) {
            return B;
        }
        return { -1 };
    }
    int start = 1;
    if (B.size() != 0) {
        start = B.back();
    }
    for (int card = start; card <= 9; card++) {
        B.push_back(card);
        vector<int> ret2 = solve(n - 1, A, B);
        if (ret2[0] != -1) {
            if (ret[0] == -1) {
                ret = vector<int>(N, 0);
                copy(ret2.begin(), ret2.end(), ret.begin());
            }
            else {
                int ret_num = get_joined_num(ret);
                int ret2_num = get_joined_num(ret2);
                if (ret_num > ret2_num) {
                    copy(ret2.begin(), ret2.end(), ret.begin());
                }
            }
        }
        B.pop_back();
    }
    return ret;
}

vector<int> solution(int n, vector<int>& A) {
    sort(A.begin(), A.end());
    vector<int> B;
    return solve(n, A, B);
}

int main() {
    int n;
    cin >> n; N = n;
    vector<int> A(n);
    for (int i = 0; i < n; i++) {
        cin >> A[i];
    }
    vector<int> B = solution(n, A);
    for (int b : B) {
        cout << b << " ";
    }
    return 0;
}
