mp = {}

applicant = []

def split_info(str) :
    s = str.split(' ')

    return [mp[s[0]], mp[s[1]], mp[s[2]], mp[s[3]], int(s[4])]

def split_query(str) :
    s = str.split(' ')

    return[mp[s[0]], mp[s[2]], mp[s[4]], mp[s[6]], int(s[7])]

def is_ok(qry, app):
    for i in range(4):
        if qry[i] != 0  and qry[i] != app[i]:
            return False
    return True

def solution(info, query):
    mp["-"] = 0

    mp["cpp"] = 1
    mp["java"] = 2
    mp["python"] = 3

    mp["backend"] = 1
    mp["frontend"] = 2

    mp["junior"] = 1
    mp["senior"] = 2

    mp["chicken"] = 1
    mp["pizza"] = 2

    for x in info:
        applicant.append(split_info(x))

    answer = []
    for q in query:
        v = split_query(q)

        cnt = 0
        for a in applicant:
            if is_ok(v, a) and v[4] <= a[4]:
                cnt += 1
        answer.append(cnt)
    return answer