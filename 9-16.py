mp = [{} for _ in range(12)]

def build_menu(src, idx, cnt, dst):
    n = len(src)
    remain = n - idx

    if remain < cnt:
        return

    if idx == n or cnt == 0:
        if cnt == 0:
            s=''.join(dst)
            if mp[len(dst)].get(s, 0) == 0:
                mp[len(s)][s] = 1
            else:
                mp[len(s)][s] += 1
        return

    build_menu(src, idx + 1, cnt, dst)

    dst.append(src[idx]);
    build_menu(src, idx + 1, cnt - 1, dst);
    dst.pop();

def solution(orders, course):
    answer = []

    for i in range(len(orders)):
        orders[i] = ''.join(sorted(orders[i]))

    for od in orders:
        for sz in course:
            s = []
            build_menu(od, 0, sz, s)

    for sz in course:
        mx = -1
        for key, value in mp[sz].items():
            mx = max(mx, value)

        if mx < 2:
            continue

        for key, value in mp[sz].items():
            if value == mx:
                answer.append(key)

    answer.sort()
    return answer