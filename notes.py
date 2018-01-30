# Charactor duplication
print ('.' * 10)

# Function call without parenthesis in python2: print "hello %s" % name
# Not any more in python3.

# Single and double quotes, they are the same.
formatter = '%r %r'
print (formatter % ("Hello world", "But it didn't sing"))

# Print parameters with '%' and ','
print ("Hello %s" % name)
print ("Hello " , name)

#  Print multiple lines of string. No wonder they say Python 2.7 is better
print ("""
    This is line 1.
    Line 2.
    """)

# What 'raw data' formatter is. The following will be printed in one line.
months = "Jan\nFeb\nMar\nApr\nMay\nJun\nJul\nAug"
print ("Months are: %s" % months)

# print call end with comma
# In python 2.7 the comma is to show that the string will be printed on the same line
# for i in range (10)
#   print i,
# will be changed in python 3 to the following:
for i in range(10)
    print (i, end = "")

# wating animation, why is there a "%s" in the end?
while True:
    for i in ["/","-","|","\\","|"]:
        print ("%s\r", i, end = "")

# raw_input function. Alright, switch to python 2.7
age = raw_input()

# function 'lambda', in the form of [param]:[return].
from heapq import nlargest
tags = [ ("python", 30), ("ruby", 25), ("c++", 50), ("lisp", 20) ]
nlargest(2, tags, key=lambda e:e[1]) # Gives [ ("c++", 50), ("python", 30) ]

# the nlargest index. use of 'enumerate'
#for item in enumerate(["a", "b", "c"]):
#    print item
#
#   (0, "a")
#   (1, "b")
#   (2, "c")
nlargest(2, enumerate(seq), key=lambda x: x[1])

# Python Dictonaries
a = {'a': 1, 'b':2}
print a
# access keys, values
print a.keys()
print a.values()

#show for loop over all entries
# option 1 using zip, this works also for iterating over any, other two lists
for k,v in zip(a.keys(), a.values()):
    print k,v

# option 2 using the dictionary `iteritems()` function
for k,v in a.iteritems():
    print k,v

