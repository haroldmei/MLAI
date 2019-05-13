

def funWithAnagrams(s):
    # Write your code here
    d = {}
    for i in range(len(s)):
        c = ''.join(sorted(s[i]))
        if c in d:
            d[c].append(i)
        else:
            d[c] = [i]
    r = []
    for i in range(len(d.values())):
        r.append(s[d.values()[i][0]])
    return r

s = ["code", "aaagmnrs", "anagrams", "doce"]
print funWithAnagrams(s)