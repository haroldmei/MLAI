#!/bin/python3

import math
import os
import random
import re
import sys
import heapq
#
# Complete the 'pickingNumbers' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER_ARRAY a as parameter.
#

def pickingNumbers(a):
    num = len(a)
    lens = [(1, 0)]*num
    for i in reversed(range(0,num - 1)):
        curmax = []
        for j in range(i + 1,num):
            if abs(a[i]-a[j]) <= 1:
                heapq.heappush(curmax, (-lens[j][0],j))
        if len(curmax) > 0:
            cur = heapq.heappop(curmax)
            curlen = list(lens[i])
            curlen[0] = 1 - cur[0]
            curlen[1] = cur[1]
            lens[i] = tuple(curlen)
    for i in range(len(lens)):
        print (i, lens[i])
        
    return lens[0]

# 4 2 3 4 4 9 98 98 3 3 3 4 2 98 1 98 98 1 1 4 98 2 98 3 9 9 3 1 4 1 98 9 9 2 9 4 2 2 9 98 4 98 1 3 4 9 1 98 98 4 2 3 98 98 1 99 9 98 98 3 98 98 4 98 2 98 4 2 1 1 9 2 4
# [(27, 2), (27, 2), (26, 3), (25, 4), (24, 8), (10, 24), (21, 7), (20, 13), (23, 9), (22, 10), (21, 12), (17, 19), (20, 14), (19, 15), (19, 17), (18, 16), (17, 20), (18, 18), (17, 21), (16, 23), (16, 22), (16, 23), (15, 30), (15, 26), (9, 25), (8, 31), (14, 28), (15, 29), (13, 35), (14, 33), (14, 39), (7, 32), (6, 34), (13, 36), (5, 38), (12, 40), (12, 37), (11, 42), (4, 45), (13, 41), (11, 43), (12, 47), (10, 46), (10, 44), (9, 49), (3, 56), (9, 50), (11, 48), (10, 52), (8, 51), (8, 51), (7, 59), (9, 53), (8, 55), (6, 64), (7, 57), (2, 70), (6, 58), (5, 60), (6, 64), (4, 61), (3, 63), (3, 66), (2, 65), (5, 67), (1, 0), (2, 72), (4, 68), (3, 69), (2, 71), (1, 0), (1, 0), (1, 0)]
# (27, 2)

if __name__ == '__main__':
    a = list(map(int, input().rstrip().split()))
    res = pickingNumbers(a)
    print(res)
