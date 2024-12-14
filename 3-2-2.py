import sys
input=sys.stdin.readline

def solution(A):
    B = ''
    i = 0
    while i < len(A):
        if (A[i] != 'a' and A[i] != 'A'):
            B += A[i]
            i += 1
            continue
        
        j = i + 1
        while j < len(A):
            if A[j] != 'a' and A[j] != 'A':
                break
            j += 1

        if j - i == 1:
            B += A[i]
        else:
            B += A[i].lower()
        i = j
        
    return B

A = input().strip()
print(solution(A))