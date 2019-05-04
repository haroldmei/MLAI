#!/bin/python3

import os
import sys
#overlapping matches
#
# Complete the bikeRacers function below.
#
def bikeRacers(bikers, bikes):
    #
    # Write your code here.
    #
    def dist(a,b):
        return (a[0] - b[0])**2 + (a[1] - b[1])**2

    dFirst = sys.maxsize
    dSecond = sys.maxsize
    i = 0
    j = 0
    First = None
    Second = None
    for i in range(len(bikers)):
        for j in range(len(bikes)):
            d = dist(bikers[i], bikes[j])
            if d < dFirst:
                First = (i, j)
                dFirst = d

    print(bikers[First[0]], bikes[First[1]])
    bikers.pop(First[0])
    bikes.pop(First[1])
    for i in range(len(bikers)):
        for j in range(len(bikes)):
            d = dist(bikers[i], bikes[j])
            if d < dSecond:
                Second = (i, j)
                dSecond = d
    
    print(bikers[Second[0]], bikes[Second[1]])
    print(dFirst,dSecond)
    return dSecond

if __name__ == '__main__':
    #fptr = open(os.environ['OUTPUT_PATH'], 'w')

    nmk = input().split()

    n = int(nmk[0])

    m = int(nmk[1])

    k = int(nmk[2])

    bikers = []

    for _ in range(n):
        bikers.append(list(map(int, input().rstrip().split())))

    bikes = []

    for _ in range(n):
        bikes.append(list(map(int, input().rstrip().split())))

    result = bikeRacers(bikers, bikes)

    #fptr.write(str(result) + '\n')

    #fptr.close()
