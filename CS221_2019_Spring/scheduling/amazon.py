

def prioritizedOrders(numOrders, orderList):
    #all Prime goes to first half
    def isNumber(userInput):
        try:
           val = int(userInput)
           return True
        except ValueError:
            return False

    for i in range(numOrders):
        orderList[i] = orderList[i].split()

    for i in range(numOrders):
        if not isNumber(orderList[i][1]):
            orderList[i][1:] = sorted(orderList[i][1:])
            orderList.insert(0,orderList.pop(i))
            last = i

    last = 0
    for i in range(numOrders):
        if not isNumber(orderList[i][1]):
            last = i

    def mycmp(s1, s2):
        if s1[0] != s2[0]:
            return cmp(s1[0], s2[0]) 
        else:
            return cmp(s1[1:], s2[1:])
    orderList[:last + 1] = sorted(orderList[:last + 1],  mycmp) 
    for i in range(numOrders):
        orderList[i] = ' '.join(orderList[i])
    return orderList

numOrders = 6
orderList = [['zld',93,12],
    ['fp','kindle','book'],
    ['1Oa','fecho','show'],
    ['1Oa','echo','show'],
    ['17g',12,25,6],
    ['abl','kindle','book'],
    ['125','echo','dot','second','generation']
]

l2 = ['al alps cow bar',
    '1Oa fecho show',
    '1Oa echo show',
    '17g 12 25 6',
    'abl kindle book',
    '125 echo dot second generation']
prioritizedOrders(numOrders, l2)
print orderList


def optimalUtilization(maxTravelDist, forwardRouteList, returnRouteList):
    forwardRouteList = sorted(forwardRouteList, lambda a,b: cmp(a[1],b[1]))
    returnRouteList = sorted(returnRouteList, lambda a,b: cmp(a[1],b[1]))
    lenf = len(forwardRouteList)
    lenr = len(returnRouteList)
    maxdist = -1
    maxflt = []
    #brute force
    for i in range(lenf):
        if forwardRouteList[i][1] >= maxTravelDist:
            break
        for j in range(lenr):
            if returnRouteList[j][1] >= maxTravelDist:
                break
            c = forwardRouteList[i][1]+returnRouteList[j][1]
            if c > maxTravelDist:
                break
            if c > maxdist:
                maxdist = c
                maxflt = [(forwardRouteList[i][0],returnRouteList[j][0])]
            elif c == maxdist:
                maxflt.append((forwardRouteList[i][0],returnRouteList[j][0]))
    return maxflt
'''
mt = 20
f = [[1,8],[2,7],[3,14]]
r = [[1,5],[2,10],[3,14]]
print optimalUtilization(mt, f,r)
'''