import sys
input=sys.stdin.readline

def solution(new_id):
    answer = ''

    new_id = new_id.lower()

    for c in new_id:
        if c.isalpha() or c.isdigit() or c in ['-', '_', '.']:
            answer += c

    while '..' in answer:
        answer = answer.replace('..', '.')

    if answer[0] == '.' and len(answer) > 1:
        answer = answer[1:]
    if answer[-1] == '.':
        answer = answer[:-1]

    if len(answer) == 0:
        answer = 'a'

    if len(answer) > 15:
        if answer[14] == '.':
            answer = answer[0:14]
        else:
            answer = answer[0:15]

    while len(answer) <=2:
        answer += answer[-1]

    return answer