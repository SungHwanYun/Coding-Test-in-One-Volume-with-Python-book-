def solution(id_list, report, k):
    n=len(id_list)

    mp = {}
    for i in range(n):
        mp[id_list[i]]=i

    report_result = [[0] * n for _ in range(n)]
    for i in range(len(report)):
        x, y = report[i].split()
        report_result[mp[x]][mp[y]] = 1

    stopped_id = [0] * n
    for j in range(n):
        cnt=0
        for i in range(n):
            cnt += report_result[i][j]
            
        if cnt >= k:
            stopped_id[j] = 1

    answer=[]
    for i in range(n): 
        cnt=0
        for j in range(n):
            if report_result[i][j] == 1 and stopped_id[j] == 1:
                cnt += 1
        answer.append(cnt)
    return answer