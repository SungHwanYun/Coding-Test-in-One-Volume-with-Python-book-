import sys
input=sys.stdin.readline

def solution(A):
    B = sorted(A)
    B = ''.join(B)
    return B
  
A = input().strip()
print(solution(A))