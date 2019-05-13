
def getUmbrellas(n, p):
    # Write your code here
    r = {}
    q = []
    for i in p:
        r[i] = 1
        q.append(i)
    
    while q:
        c = q.pop(0)
        for i in p:
            if i + c == n:
                if i+c in r:
                    return r[i+c]
                else:
                    return r[c] + 1
            elif i + c > n:
                if c in r:
                    return r[c]
                else:
                    return -1

            if i + c not in r:
                r[i + c] = r[c] + 1
                q.append(i+c)
    return -1

n = 4
p = [2,2,4]

print getUmbrellas(n,p)