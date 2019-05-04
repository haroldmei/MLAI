import math
import os
import random
import re
import sys

# Complete the matrixRotation function below.
def matrixRotation(matrix, r):
    m = len(matrix)
    n = len(matrix[0])
    num_rect = int(min(m,n)/2)
    for num in range(num_rect):
        curr = []
        for j in range(num, m - num - 1):
            curr.append(matrix[j][num]) # left
        for j in range(num, n - num - 1):
            curr.append(matrix[m - num - 1][j]) # bottom
        for j in reversed(range(num + 1, m - num)):
            curr.append(matrix[j][n - num - 1])
        for j in reversed(range(num + 1, n - num)):
            curr.append(matrix[num][j])

        rr = r % len(curr)
        for i in range(rr):
            curr.insert(0, curr.pop())

        for j in range(num, m - num - 1):
            matrix[j][num] = curr.pop(0)
        for j in range(num, n - num - 1):
            matrix[m - num - 1][j] = curr.pop(0)
        for j in reversed(range(num + 1, m - num)):
            matrix[j][n - num - 1] = curr.pop(0)
        for j in reversed(range(num + 1, n - num)):
            matrix[num][j] = curr.pop(0)

    strs = '\n'.join([' '.join(str(e) for e in matrix[i]) for i in range(m)])
    print(strs)
    
if __name__ == '__main__':
    mnr = input().rstrip().split()

    m = int(mnr[0])

    n = int(mnr[1])

    r = int(mnr[2])

    matrix = []

    for _ in range(m):
        matrix.append(list(map(int, input().rstrip().split())))

    matrixRotation(matrix, r)
