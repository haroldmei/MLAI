
import math
import os
import random
import re
import sys

def add(a, b):
    result = []
    carry = 0
    for op1, op2 in zip(reversed(a), reversed(b)):
        cur = int(op1) + int(op2) + carry
        carry = 0
        if cur >= 10:
            carry = 1
            cur -= 10
        result.insert(0, str(cur))
    res = ''.join(result)
    return res
    
def Product(a,b):
    result = []
    for i in reversed(b):
        carry = 0
        for j in reversed(a):
            

res = add('192919293949596', '122121213152341234163241234')
print res