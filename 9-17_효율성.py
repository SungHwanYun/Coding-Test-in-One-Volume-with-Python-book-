import bisect

score = [[[[[] for _ in range(4)] for _ in range(3)] for _ in range(3)] for _ in range(3)]

mp = {}

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

    for v in info:
        x = split_info(v)
        for lang in range(4):
            for job in range(3):
                for career in range(3):
                    for food in range(3):
                        y = [lang, job, career, food]
                        if is_ok(y, x):
                            score[food][career][job][lang].append(int(x[4]));
    
    for lang in range(4):
        for job in range(3):
            for career in range(3):
                for food in range(3):                
                    score[food][career][job][lang].sort()

    answer = []
    for v in query:
        q = split_query(v)
        lang = q[0]; job = q[1]; car = q[2]; food = q[3]; point = q[4];
        x = len(score[food][car][job][lang]) - bisect.bisect_left(score[food][car][job][lang], point)
        answer.append(x)
    return answer