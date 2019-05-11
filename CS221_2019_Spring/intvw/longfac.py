
#!/bin/python

import math
import os
import random
import re
import sys

# Complete the extraLongFactorials function below.
def extraLongFactorials(n):
    def _add(a,b):
        la = len(a)
        lb = len(b)
        # fill 0s
        if la > lb:
            for i in range(la - lb):
                b.insert(0,0)
        elif la < lb:
            for i in range(lb - la):
                a.insert(0,0)
        result = [0]*max(la,lb)
        c = 0
        for i in reversed(range(lb)):
            p1 = a[i]
            p2 = b[i]
            r = p1 + p2 + c
            c = int(r / 10)
            result[i] = int(r % 10)

        if c > 0:
            result.insert(0,c)
        return result
    
    # multiply with a single digit
    def __mul(a,b):
        c = 0
        la = len(a)
        result = [0]*la
        for i in reversed(range(la)):
            r = a[i] * b + c
            c = int(r / 10)
            result[i] = int(r % 10)
        if c > 0:
            result.insert(0, c)
        return result
    def _mul(a,b):
        lb = len(b)
        r = [0]
        for i in reversed(range(lb)):
            cur = __mul(a, b[i])
            if lb - i > 1:
                cur = cur + [0]*(lb - i - 1)    # insert 0s
            r = _add(r, cur)
        return r
    def intToList(r):
        op1 = []
        while r > 0:
            op1.insert(0, r%10)
            r = int(r / 10)
        return op1
    def listToInt(a):
        l = len(a)
        r = 0
        for i in reversed(range(l)):
            r = r + a[i] * (10**(l - i - 1))
        return r
    r = [1]
    for i in range(n):
        #op1 = intToList(r)
        op2 = intToList(i + 1)
        r = _mul(r, op2)
    return listToInt(r)


def f(n):
    r = 1
    for i in range(n):
        r = r * (i+1)
    return r

print(f(1000))
print(extraLongFactorials(1000))