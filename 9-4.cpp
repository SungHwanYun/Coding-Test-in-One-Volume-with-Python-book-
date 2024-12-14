#include <bits/stdc++.h>
using namespace std;

vector<int> translate_binary(long long n) {
    vector<int> answer;
    while (n > 0) {
        int x = n % 2;
        n = n / 2;
        answer.push_back(x);
    }
    long long y = 1;
    while (y <= answer.size()) {
        y = y * 2;
    }
    while (answer.size() + 1 < y) {
        answer.push_back(0);
    }
    reverse(answer.begin(), answer.end());
    return answer;
}

int solve(vector<int>& b, int st, int en) {
    if (st == en) {
        return 1;
    }
    int r = (en + st) / 2;
    if (b[r] == 0) {
        for (int i = st; i <= en; i++) {
            if (b[i] == 1) {
                return 0;
            }
        }
        return 1;
    }
    else {
        return solve(b, st, r - 1) & solve(b, r + 1, en);
    }
}

vector<int> solution(vector<long long> numbers) {
    vector<int> answer;
    for (long long n : numbers) {
        vector<int> b = translate_binary(n);
        answer.push_back(solve(b, 0, b.size() - 1));
    }
    return answer;
}